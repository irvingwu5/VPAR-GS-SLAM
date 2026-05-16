"""
Error-mask-based Gaussian densification replacing gradient-based clone/split.
Adapted from FGS-SLAM for 2DGS surfels.

Determines where to add new Gaussians based on rendering quality:
  1. Silhouette error (low opacity = missing surfaces)
  2. Depth error (rendered behind GT, error > threshold * median)
  3. RGB error (per-pixel L1 sum exceeds threshold)
"""

import torch
import torch.nn.functional as F

from gaussian_splatting.utils.general_utils import inverse_sigmoid, normal2rotation
from gaussian_splatting.utils.sh_utils import RGB2SH


class ErrorMaskDensifier:
    """
    Error-driven Gaussian creation replacing gradient-based clone/split.

    When enabled, the BackEnd:
      1. Renders each keyframe in the window
      2. Computes three error masks (silhouette, depth, RGB)
      3. OR-combines them, intersects with FFT masks
      4. Unprojects depth -> 3D points -> creates new 2DGS surfels
    """

    def __init__(self, config, gaussian_model):
        cfg = config.get("errormask", {}) if isinstance(config, dict) else {}
        self.enabled = cfg.get("enabled", False)

        self.silhouette_thresh = cfg.get("silhouette_thresh", 0.98)
        self.depth_error_mult = cfg.get("depth_error_mult", 10.0)
        self.rgb_error_thresh = cfg.get("rgb_error_thresh", 0.5)

        self.min_new_gaussians = cfg.get("min_new_gaussians", 0)
        self.max_new_gaussians = cfg.get("max_new_gaussians", 5000)

        self.new_scale_min = cfg.get("new_scale_min", 0.005)
        self.new_scale_max = cfg.get("new_scale_max", 0.1)
        self.new_opacity = cfg.get("new_opacity", 0.5)

        self.apply_every = cfg.get("apply_every", 100)
        self.apply_offset = cfg.get("apply_offset", 50)

        self.gaussian_model = gaussian_model

    def compute_error_mask(self, render_pkg, viewpoint):
        """
        Compute combined error mask from rendering output.

        Args:
            render_pkg: dict with keys 'render' (3,H,W), 'depth' (1,H,W), 'opacity' (1,H,W)
            viewpoint: Camera with .original_image (3,H,W), .depth (H,W numpy)
        Returns:
            error_mask: (H, W) bool
            error_dict: {'silhouette': (H,W), 'depth': (H,W), 'rgb': (H,W)} for debugging
        """
        device = render_pkg['render'].device
        H, W = viewpoint.image_height, viewpoint.image_width

        gt_image = viewpoint.original_image.cuda()
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=device
        ).unsqueeze(0)

        rendered_image = render_pkg['render']
        rendered_depth = render_pkg['depth']
        rendered_opacity = render_pkg['opacity']

        # 1. Silhouette error: incomplete surface coverage
        silhouette_error = rendered_opacity < self.silhouette_thresh

        # 2. Depth error: rendered behind GT AND error exceeds threshold * median
        valid_depth = gt_depth > 0.01
        depth_diff = rendered_depth - gt_depth
        depth_error_abs = torch.abs(depth_diff)

        depth_error = torch.zeros_like(valid_depth, dtype=torch.bool)
        if valid_depth.sum() > 0:
            median_error = depth_error_abs[valid_depth].median()
            threshold = median_error * self.depth_error_mult
            depth_error = (depth_diff > 0) & (depth_error_abs > threshold) & valid_depth

        # 3. RGB error: per-pixel L1 sum across channels
        rgb_diff = torch.abs(rendered_image - gt_image).sum(dim=0, keepdim=True)
        rgb_error = rgb_diff > self.rgb_error_thresh

        # OR-combine
        error_mask = (silhouette_error | depth_error | rgb_error).squeeze(0)

        return error_mask, {
            'silhouette': silhouette_error.squeeze(0),
            'depth': depth_error.squeeze(0),
            'rgb': rgb_error.squeeze(0),
        }

    def select_pixels(self, error_mask, fft_masks):
        """
        Select pixel positions from error mask, constrained by FFT masks.

        Density control via FFT masks:
          - opacity_mask (True=high-freq): only place in textured areas
          - high_freq_mask: dense grid for high-freq pixels
          - low_freq_mask: sparse grid for low-freq pixels

        Args:
            error_mask: (H, W) bool
            fft_masks: dict from FFTMaskGenerator.compute_masks(), or None
        Returns:
            (N,) long tensor of flat indices into (H*W)
        """
        if fft_masks is None:
            selected = error_mask
        else:
            opacity = fft_masks.get('opacity_mask')
            high_freq = fft_masks.get('high_freq_mask')
            low_freq = fft_masks.get('low_freq_mask')
            if opacity is None:
                selected = error_mask
            else:
                selected_high = error_mask & opacity & high_freq
                selected_low = error_mask & (~opacity) & low_freq
                selected = selected_high | selected_low

        flat = selected.flatten().nonzero(as_tuple=True)[0]

        if len(flat) > self.max_new_gaussians:
            perm = torch.randperm(len(flat), device=flat.device)[:self.max_new_gaussians]
            flat = flat[perm]

        return flat

    def create_gaussians_at_pixels(self, viewpoint, pixel_indices, fft_masks, cur_frame_idx):
        """
        Create new 2DGS surfels at selected pixel positions using GT depth for unprojection.

        Args:
            viewpoint: Camera with .depth (H, W numpy) for GT depth
            pixel_indices: (N,) long, flat indices into (H*W)
            fft_masks: dict from FFTMaskGenerator, or None
            cur_frame_idx: int, keyframe ID for the new Gaussians
        Returns:
            bool: True if any Gaussians were created
        """
        if len(pixel_indices) < self.min_new_gaussians:
            return False

        H, W = viewpoint.image_height, viewpoint.image_width
        device = pixel_indices.device

        # Use GT depth for unprojection (places Gaussians at true surface positions)
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=device
        )

        rows = pixel_indices // W
        cols = pixel_indices % W

        selected_depth = gt_depth[rows, cols]

        valid_depth = selected_depth > 0.01
        if valid_depth.sum() == 0:
            return False

        pixel_indices = pixel_indices[valid_depth]
        rows = rows[valid_depth]
        cols = cols[valid_depth]
        selected_depth = selected_depth[valid_depth]
        N = len(pixel_indices)

        # ---- Unproject to 3D world coordinates ----
        c2w = torch.inverse(viewpoint.world_view_transform.T)
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]

        fx, fy = viewpoint.fx, viewpoint.fy
        cx, cy = viewpoint.cx, viewpoint.cy
        ray_x = (cols.float() - cx) / fx
        ray_y = (rows.float() - cy) / fy
        rays_d = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)
        rays_d = rays_d @ R_c2w.T
        rays_d = F.normalize(rays_d, dim=-1)

        new_xyz = t_c2w.unsqueeze(0) + selected_depth.unsqueeze(-1) * rays_d

        # ---- Color from GT image ----
        gt_image = viewpoint.original_image.cuda()
        new_rgb = gt_image[:, rows, cols].T

        # ---- Scales (frequency-adaptive from FFT, or depth-based default) ----
        if fft_masks is not None and fft_masks.get('scale_map') is not None:
            pixel_scales = fft_masks['scale_map'][rows, cols]
            world_scale = pixel_scales * self.new_scale_max
        else:
            focal_avg = (fx + fy) / 2.0
            world_scale = selected_depth / focal_avg * 0.01

        world_scale = torch.clamp(world_scale, min=self.new_scale_min, max=self.new_scale_max)
        new_scaling = torch.log(world_scale)[:, None].repeat(1, 2)

        # ---- Opacity ----
        new_opacity = inverse_sigmoid(torch.full((N, 1), self.new_opacity, device=device))

        # ---- Normals and Rotation ----
        new_normals = self._estimate_normals_from_depth(gt_depth, rows, cols)
        new_rotation = normal2rotation(new_normals)

        # ---- SH features (format: (N, coeff, channels)) ----
        sh_features = RGB2SH(new_rgb)  # (N, 3)
        new_features_dc = sh_features.unsqueeze(1)  # (N, 1, 3)
        max_sh = self.gaussian_model.max_sh_degree
        new_features_rest = torch.zeros(N, (max_sh + 1) ** 2 - 1, 3, device=device)  # (N, 0, 3)

        # ---- Add to GaussianModel ----
        self.gaussian_model.densification_postfix(
            new_xyz=new_xyz,
            new_features_dc=new_features_dc,
            new_features_rest=new_features_rest,
            new_opacities=new_opacity,
            new_scaling=new_scaling,
            new_rotation=new_rotation,
            new_kf_ids=torch.full((N,), cur_frame_idx, dtype=torch.int, device='cpu'),
            new_n_obs=torch.zeros(N, dtype=torch.int, device='cpu'),
        )

        return True

    @staticmethod
    def _estimate_normals_from_depth(depth, rows, cols, neighbor_radius=2):
        """
        Estimate surface normals from depth neighborhood via finite differences.

        Args:
            depth: (H, W) float32
            rows, cols: (N,) pixel coordinates
            neighbor_radius: pixels
        Returns:
            (N, 3) normalized normals in camera frame (z-forward)
        """
        H, W = depth.shape
        d_pad = F.pad(depth.unsqueeze(0), (neighbor_radius, neighbor_radius,
                       neighbor_radius, neighbor_radius), mode='replicate')[0]

        r_shifted = rows + neighbor_radius
        c_shifted = cols + neighbor_radius

        d_left = d_pad[r_shifted, c_shifted - neighbor_radius]
        d_right = d_pad[r_shifted, c_shifted + neighbor_radius]
        d_up = d_pad[r_shifted - neighbor_radius, c_shifted]
        d_down = d_pad[r_shifted + neighbor_radius, c_shifted]

        dx = d_right - d_left
        dy = d_down - d_up

        normals = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=-1)
        normals = F.normalize(normals, dim=-1)

        return normals
