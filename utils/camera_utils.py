import torch
from torch import nn

import cv2
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
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
        kf_id=-1,
        prior_mono_depth=None
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.video_idx = kf_id
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color  # [C, H, W]
        self.depth = depth # [H, W]
        self.mono_depth = prior_mono_depth # unscaled prior depth from a monocular depth prediction network
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        # Pyramid training member variables
        self.pyramid_original_image = [] # levels of down-sampled image of the original image, the first one has the lowest resolution.
        self.pyramid_times_of_use = [] # a list of integers
        self.num_pyramid_sub_levels = 0 # total number of levels in the pyramid
        self.pyramid_width = [] # widths of pyramid levels
        self.pyramid_height = [] # heights of pyramid levels
        self.remaining_times_of_use = 0

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
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        """
        R: world frame orientation relative to the camera frame.
        t: world origin position in the camera frame.
        """
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def c2w(self):
        """pose of the camera wrt the world frame"""
        Rt = torch.zeros((4, 4), device=self.device)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        Rt[3, 3] = 1.0
        C2W = torch.linalg.inv(Rt)
        return C2W

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

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None

    # Pyramid training methods
    def get_current_pyramid_level(self):
        for i in range(len(self.pyramid_times_of_use)):
            if self.pyramid_times_of_use[i] > 0:
                self.pyramid_times_of_use[i] -= 1
                return i
        # If all sub-levels have been used up
        return self.num_pyramid_sub_levels

    def build_pyramid(self, num_pyramid_sub_levels, pyramid_times_of_use,
                    pyramid_height, pyramid_width):
        self.num_pyramid_sub_levels = num_pyramid_sub_levels
        self.pyramid_times_of_use = pyramid_times_of_use
        self.pyramid_height = pyramid_height
        self.pyramid_width = pyramid_width
        # Assuming original_image is in [C, H, W] format and values in [0, 1]
        np_image = self.original_image.permute(1, 2, 0).cpu().numpy()  # Convert [C, H, W] -> [H, W, C] and NumPy
        self.pyramid_original_image = [
            torch.tensor(cv2.resize(np_image,(self.pyramid_width[l], self.pyramid_height[l]))).permute(2, 0, 1)  # Convert back to [C, H, W]
            for l in range(self.num_pyramid_sub_levels)]

    def increase_times_of_use(self, newlife):
        self.remaining_times_of_use += newlife