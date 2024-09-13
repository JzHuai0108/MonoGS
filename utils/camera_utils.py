import torch
from torch import nn
import torch.nn.functional as F

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_splatting.utils.general_utils import build_rotation
from utils.pose_utils import matrix_to_quaternion
from utils.slam_utils import image_gradient, image_gradient_mask


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.q_cw = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, requires_grad=True, device=device))
        self.p_cw = nn.Parameter(
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, requires_grad=True, device=device))

        if uid == 0:
            self.q_cw.requires_grad = False
            self.p_cw.requires_grad = False
            print("Freezing pose of camera at uid 0")

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def full_proj_transform_updated(self):
        return (
            self.world_view_transform_updated.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def world_view_transform(self):
        """
        return transposed world to camera transform
        """
        return self.world_view_transform_updated

    @property
    def world_view_transform_updated(self):
        """
        return transposed world to camera transform
        """
        T_w2c = torch.eye(4, device=self.device).float()
        q_w2c = self.q_cw.unsqueeze(0)
        T_w2c[0:3, 0:3] = build_rotation(q_w2c)
        T_w2c[0:3, 3] = self.p_cw
        return T_w2c.transpose(-2, -1)

    @property
    def world_view_rotation_updated(self):
        """
        return updated world to camera rotation
        """
        q_w2c = self.q_cw.unsqueeze(0)
        return build_rotation(q_w2c)

    @property
    def camera_center_updated(self):
        T_cw_transposed = self.world_view_transform_updated
        cam_center = (-T_cw_transposed[3, :3]) @ T_cw_transposed[:3, :3].transpose(-2, -1)
        return cam_center.squeeze()

    @property
    def camera_center_updated2(self):
        T_cw_transposed = self.world_view_transform_updated
        return T_cw_transposed.inverse()[3, :3]

    @property
    def world_view_transform_camcentric(self):
        return torch.eye(4, device=self.device)

    @property
    def camera_center_camcentric(self):
        return torch.zeros(3, device=self.device)

    @property
    def full_proj_transform_camcentric(self):
        return self.projection_matrix

    def update_qp(self, q, t):
        norm_q = F.normalize(q, p=2, dim=-1)
        self.q_cw.data = norm_q.float().to(device=self.device)
        self.p_cw.data = t.float().to(device=self.device)

    def update_RT(self, R, t):
        self.q_cw.data = matrix_to_quaternion(R).float().to(device=self.device)
        self.p_cw.data = t.float().to(device=self.device)

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.exposure_a = None
        self.exposure_b = None
