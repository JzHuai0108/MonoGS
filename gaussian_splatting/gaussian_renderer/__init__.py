#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from diff_plane_rasterization import (
    GaussianRasterizationSettings as GaussianPlaneRasterizationSettings,
    GaussianRasterizer as GaussianPlaneRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.utils.graphics_utils import normal_from_depth_image


# render_normal is copied from PGSR gaussian_renderer/__init__.py
def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale],
                                            intrinsic_matrix.to(depth.device),
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if mask is not None:
        rendered_image, radii, depth, opacity = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask],
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }


def plane_render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    return_plane=False,
    return_depth_normal=False
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianPlaneRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=return_plane,
        debug=False,
    )

    rasterizer = GaussianPlaneRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # check if the covariance is isotropic
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            # we do not use camera_center_updated here because the gradient of
            # color to pose is considered in cuda computeColorFromSH().
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    if not return_plane:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if mask is not None:
            rendered_image, radii, depth, opacity, n_touched, _, _ = rasterizer(
                means3D=means3D[mask],
                means2D=means2D[mask],
                means2D_abs=means2D_abs[mask],
                shs=shs[mask],
                colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
                opacities=opacity[mask],
                scales=scales[mask],
                rotations=rotations[mask],
                cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
                theta=viewpoint_camera.cam_rot_delta,
                rho=viewpoint_camera.cam_trans_delta,
            )
        else:
            rendered_image, radii, depth, opacity, n_touched, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                means2D_abs=means2D_abs,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                theta=viewpoint_camera.cam_rot_delta,
                rho=viewpoint_camera.cam_trans_delta,
            )
        return_dict = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_abs": screenspace_points_abs,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "opacity": opacity,
            "n_touched": n_touched,
        }
        return return_dict
    else:
        global_normal = pc.get_normal(viewpoint_camera)
        local_normal = global_normal @ viewpoint_camera.world_view_transform_updated[:3, :3]
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform_updated[:3, :3] + viewpoint_camera.world_view_transform_updated[3, :3]
        depth_z = pts_in_cam[:, 2]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        if mask is not None:
            rendered_image, radii, depth, opacity, n_touched, out_all_map, plane_depth = rasterizer(
                means3D=means3D[mask],
                means2D=means2D[mask],
                means2D_abs=means2D_abs[mask],
                shs=shs[mask],
                colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
                opacities=opacity[mask],
                scales=scales[mask],
                rotations=rotations[mask],
                cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
                theta=viewpoint_camera.cam_rot_delta,
                rho=viewpoint_camera.cam_trans_delta,
                all_map=input_all_map[mask])
        else:
            rendered_image, radii, depth, opacity, n_touched, out_all_map, plane_depth = rasterizer(
                means3D=means3D,
                means2D=means2D,
                means2D_abs=means2D_abs,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                theta=viewpoint_camera.cam_rot_delta,
                rho=viewpoint_camera.cam_trans_delta,
                all_map=input_all_map)
        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4]
        rendered_distance = out_all_map[4:5]
        return_dict = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_abs": screenspace_points_abs,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "opacity": opacity,
            "n_touched": n_touched,
            "rendered_normal": rendered_normal,
            "plane_depth": plane_depth,
            "rendered_distance": rendered_distance,
        }

        if return_depth_normal:
            depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * rendered_alpha.detach()
            return_dict.update({"depth_normal": depth_normal})

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return return_dict
