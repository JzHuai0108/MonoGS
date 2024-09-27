import numpy as np
import torch
import torch.nn.functional as F
from gaussian_splatting.utils.loss_utils import get_img_grad_weight, get_points_depth_in_depth_map, get_points_from_depth, lncc
from gaussian_splatting.utils.image_utils import erode
from gaussian_splatting.utils.graphics_utils import patch_offsets, patch_warp


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_planar_multiview_loss_mapping(config, plane_depth, nearest_plane_depth, rendered_normal, rendered_distance,
                                      viewpoint_cam, nearest_cam):
    patch_size = config['Training']['multi_view_patch_size']
    sample_num = config['Training']['multi_view_sample_num']
    pixel_noise_th = config['Training']['multi_view_pixel_noise_th']
    total_patch_size = (patch_size * 2 + 1) ** 2
    ncc_weight = config['Training']['multi_view_ncc_weight']
    geo_weight = config['Training']['multi_view_geo_weight']
    ## compute geometry consistency mask and loss
    H, W = plane_depth.squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(plane_depth.device)

    ref_T_W2C_transposed = viewpoint_cam.world_view_transform_updated
    ref_R_C2W = ref_T_W2C_transposed[:3, :3].clone().float().cuda()
    ref_p_W2C = ref_T_W2C_transposed[3, :3].clone().float().cuda()

    nearest_T_W2C_transposed = nearest_cam.world_view_transform_updated
    nearest_R_C2W = nearest_T_W2C_transposed[:3, :3].clone().float().cuda()
    nearest_p_W2C = nearest_T_W2C_transposed[3, :3].clone().float().cuda()

    pts = get_points_from_depth(viewpoint_cam, plane_depth)
    pts_in_nearest_cam = pts @ nearest_R_C2W + nearest_p_W2C
    map_z, d_mask = get_points_depth_in_depth_map(nearest_cam, nearest_plane_depth,
                                                            pts_in_nearest_cam)

    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
    pts_ = (pts_in_nearest_cam - nearest_p_W2C) @ nearest_R_C2W.transpose(-1, -2)
    pts_in_view_cam = pts_ @ ref_R_C2W + ref_p_W2C
    pts_projections = torch.stack(
        [pts_in_view_cam[:, 0] * viewpoint_cam.fx / pts_in_view_cam[:, 2] + viewpoint_cam.cx,
         pts_in_view_cam[:, 1] * viewpoint_cam.fy / pts_in_view_cam[:, 2] + viewpoint_cam.cy], -1).float()
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
    d_mask = d_mask & (pixel_noise < pixel_noise_th)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~d_mask] = 0

    geo_loss = 0
    ncc_loss = 0
    ncc_scale = 1.0
    if d_mask.sum() > 0:
        geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
        with (torch.no_grad()):
            ## sample mask
            d_mask = d_mask.reshape(-1)
            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
            if d_mask.sum() > sample_num:
                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                valid_indices = valid_indices[index]

            weights = weights.reshape(-1)[valid_indices]
            ## sample ref frame patch
            pixels = pixels.reshape(-1, 2)[valid_indices]
            offsets = patch_offsets(patch_size, pixels.device)
            ori_pixels_patch = pixels.reshape(-1, 1, 2) / ncc_scale + offsets.float()

            gt_image_gray = viewpoint_cam.gray_image.cuda()
            H, W = gt_image_gray.squeeze().shape
            pixels_patch = ori_pixels_patch.clone()
            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                         align_corners=True)
            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

            ref_to_nearest_r = nearest_R_C2W.transpose(-1, -2) @ ref_R_C2W
            ref_to_nearest_t = -ref_to_nearest_r @ ref_p_W2C + nearest_p_W2C

        ## compute Homography
        ref_local_n = rendered_normal.permute(1, 2, 0)
        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

        ref_local_d = rendered_distance.squeeze()
        # rays_d = viewpoint_cam.get_rays()
        # rendered_normal2 = rendered_normal.reshape(-1,3)
        # ref_local_d = plane_depth.view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
        # ref_local_d = ref_local_d.reshape(H,W)

        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
        H_ref_to_nearest = ref_to_nearest_r[None] - \
                            torch.matmul(ref_to_nearest_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                         ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / ref_local_d[..., None, None]
        H_ref_to_nearest = torch.matmul(
            nearest_cam.get_K(ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
            H_ref_to_nearest)
        H_ref_to_nearest = H_ref_to_nearest @ viewpoint_cam.get_inv_K(ncc_scale)

        ## compute nearest frame patch
        grid = patch_warp(H_ref_to_nearest.reshape(-1, 3, 3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
        nearest_image_gray = nearest_cam.gray_image.cuda()
        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                         align_corners=True)
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

        ## compute loss
        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
        mask = ncc_mask.reshape(-1)
        ncc = ncc.reshape(-1) * weights
        ncc = ncc[mask].squeeze()

        if mask.sum() > 0:
            ncc_loss = ncc_weight * ncc.mean()

    return geo_loss + ncc_loss


def get_planar_loss_mapping(config, image, depth, rendered_normal, depth_normal,
                            viewpoint, visible_scales, enable_normal_loss, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image_ab * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    rgb_loss = l1_rgb.mean()

    if visible_scales is not None and visible_scales.numel() > 0:
        sorted_scale, _ = torch.sort(visible_scales, dim=-1)
        min_scale_loss = sorted_scale[..., 0].mean() * 100
    else:
        min_scale_loss = 0

    if enable_normal_loss:
        # single-view loss
        weight = config["Training"]["single_view_weight"]
        image_weight = 1.0 - get_img_grad_weight(gt_image)
        image_weight = image_weight.clamp(0, 1).detach() ** 5
        image_weight = erode(image_weight[None, None]).squeeze()
        normal_loss = weight * (image_weight * ((depth_normal - rendered_normal).abs().sum(0))).mean()
    else:
        normal_loss = 0

    if config["Training"]["monocular"]:
        return rgb_loss + min_scale_loss + normal_loss

    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image_ab.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*mask_shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    depth_loss = l1_depth.mean()
    return alpha * rgb_loss + (1 - alpha) * depth_loss + min_scale_loss + normal_loss


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
