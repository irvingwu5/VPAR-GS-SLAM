import torch
import torch.nn.functional as F


def intrins_to_intrins_inv(intrins):
    """intrins to intrins_inv

    NOTE: top-left is (0,0)
    """
    if torch.is_tensor(intrins):
        intrins_inv = torch.zeros_like(intrins)
    else:
        raise Exception("intrins should be torch tensor or numpy array")

    intrins_inv[0, 0] = 1 / intrins[0, 0]
    intrins_inv[0, 2] = -intrins[0, 2] / intrins[0, 0]
    intrins_inv[1, 1] = 1 / intrins[1, 1]
    intrins_inv[1, 2] = -intrins[1, 2] / intrins[1, 1]
    intrins_inv[2, 2] = 1.0
    return intrins_inv


def get_cam_coords(intrins_inv, depth):
    """camera coordinates from intrins_inv and depth

    NOTE: intrins_inv should be a torch tensor of shape (B, 3, 3)
    NOTE: depth should be a torch tensor of shape (B, 1, H, W)
    NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins_inv) and intrins_inv.ndim == 3
    assert torch.is_tensor(depth) and depth.ndim == 4
    assert intrins_inv.dtype == depth.dtype
    assert intrins_inv.device == depth.device
    B, _, H, W = depth.size()

    u_range = (
        torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, W).expand(H, W)
    )  # (H, W)
    v_range = (
        torch.arange(H, dtype=depth.dtype, device=depth.device).view(H, 1).expand(H, W)
    )  # (H, W)
    ones = torch.ones(H, W, dtype=depth.dtype, device=depth.device)
    pixel_coords = (
        torch.stack((u_range, v_range, ones), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    )  # (B, 3, H, W)
    pixel_coords = pixel_coords.view(B, 3, H * W)  # (B, 3, H*W)

    cam_coords = intrins_inv.bmm(pixel_coords).view(B, 3, H, W)
    cam_coords = cam_coords * depth
    return cam_coords


def d2n_tblr(
    points: torch.Tensor, k: int = 3, d_min: float = 1e-3, d_max: float = 10.0
) -> torch.Tensor:
    """points:     3D points in camera coordinates, shape: (B, 3, H, W)
    k:          neighborhood size
    d_min/max:  range of valid depth values
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(
        points, (k, k, k, k), mode="constant", value=0
    )  # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
        points_pad[:, 2:, :, :] < d_max
    )  # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = (
        points_pad[:, :, :H, k : k + W] - points_pad[:, :, 2 * k : 2 * k + H, k : k + W]
    )

    # horizontal vector (left - right)
    vec_hori = (
        points_pad[:, :, k : k + H, :W] - points_pad[:, :, k : k + H, 2 * k : 2 * k + W]
    )

    # valid_mask (all five depth values should be valid)
    valid_mask = (
        valid_pad[:, :, k : k + H, k : k + W]
        * valid_pad[:, :, :H, k : k + W]
        * valid_pad[:, :, 2 * k : 2 * k + H, k : k + W]
        * valid_pad[:, :, k : k + H, :W]
        * valid_pad[:, :, k : k + H, 2 * k : 2 * k + W]
    )
    valid_mask = valid_mask > 0.5

    # cross product
    cross_product = -torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)

    return normal, valid_mask
