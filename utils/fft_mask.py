"""
FFT-based frequency analysis for adaptive Gaussian placement density and scale.
Adapted from FGS-SLAM for 2DGS (2-component scaling).
"""

import torch
import torch.nn.functional as F


class FFTMaskGenerator:
    """
    Generates frequency-based masks controlling where Gaussians are placed
    and at what scale, based on local image texture content.

    Pipeline:
      1. RGB -> grayscale -> CLAHE contrast enhancement
      2. 2D FFT -> log-magnitude spectrum
      3. Multi-band Gaussian high-pass filters -> accumulated spectrogram
      4. Triangle threshold -> binary opacity_mask (True = high-freq, dense placement)
      5. Grid sampling masks at two densities (high_freq_stride, low_freq_stride)
      6. Per-pixel scale map (small in high-freq, large in low-freq)
    """

    def __init__(self, config):
        cfg = config.get("fftmask", {}) if isinstance(config, dict) else {}
        self.enabled = cfg.get("enabled", False)

        self.clip_limit = cfg.get("clahe_clip_limit", 2.0)
        self.tile_grid_size = tuple(cfg.get("clahe_tile_grid", [8, 8]))

        self.num_bands = cfg.get("num_bands", 4)
        self.base_sigma = cfg.get("base_sigma", 5.0)
        self.sigma_step = cfg.get("sigma_step", 5.0)
        self.log_epsilon = cfg.get("log_epsilon", 1e-6)

        self.triangle_bins = cfg.get("triangle_bins", 256)

        self.high_freq_stride = cfg.get("high_freq_stride", 2)
        self.low_freq_stride = cfg.get("low_freq_stride", 4)

        self.scale_min = cfg.get("scale_min", 0.01)
        self.scale_max = cfg.get("scale_max", 0.1)
        self.depth_scale_factor = cfg.get("depth_scale_factor", 1.0)

    @staticmethod
    def _clahe_torch(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Simplified CLAHE in PyTorch (no OpenCV dependency).

        Args:
            img: (H, W) float32 tensor in [0, 1]
            clip_limit: contrast enhancement clip limit
            tile_grid_size: (tile_h, tile_w) number of tiles
        Returns:
            (H, W) float32 tensor in [0, 1]
        """
        H, W = img.shape
        th, tw = tile_grid_size
        bh = H // th
        bw = W // tw

        pad_h = (bh * th) - H
        pad_w = (bw * tw) - W
        if pad_h > 0 or pad_w > 0:
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        else:
            img_pad = img

        tiles = img_pad.reshape(th, bh, tw, bw).permute(0, 2, 1, 3).reshape(th * tw, bh, bw)

        result_tiles = []
        for t in range(th * tw):
            tile = tiles[t].flatten()
            sorted_vals, sort_idx = torch.sort(tile)
            cdf = torch.arange(1, len(sorted_vals) + 1, device=img.device, dtype=torch.float32)
            cdf = cdf / cdf[-1]

            hist = torch.diff(
                torch.cat([torch.zeros(1, device=img.device),
                           cdf[sort_idx.argsort()]])
            )
            clip_val = clip_limit * hist.mean()
            excess = torch.clamp(hist - clip_val, min=0)
            hist = torch.clamp(hist, max=clip_val) + excess.sum() / len(hist)

            cdf = torch.cumsum(hist, dim=0)
            cdf = cdf / cdf[-1]
            equalized = cdf[sort_idx.argsort()].reshape(bh, bw)
            result_tiles.append(equalized)

        result = torch.stack(result_tiles).reshape(th, tw, bh, bw).permute(0, 2, 1, 3).reshape(th * bh, tw * bw)
        return result[:H, :W]

    def generate_frequency(self, rgb_image):
        """
        Convert RGB to frequency-domain log-magnitude spectrum.

        Args:
            rgb_image: (3, H, W) float32 in [0, 1]
        Returns:
            (H, W) float32 log-magnitude FFT spectrum
        """
        gray = rgb_image.mean(dim=0)
        gray_enhanced = self._clahe_torch(gray, self.clip_limit, self.tile_grid_size)

        fft = torch.fft.fft2(gray_enhanced)
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted)
        log_magnitude = torch.log(magnitude + self.log_epsilon)

        return log_magnitude

    def multiLayer_spectrogram(self, fft_spectrum):
        """
        Multi-band Gaussian high-pass filtering + Triangle thresholding.

        Args:
            fft_spectrum: (H, W) float32 log-magnitude FFT spectrum
        Returns:
            opacity_mask: (H, W) bool, True = high-frequency (dense placement)
        """
        H, W = fft_spectrum.shape
        cy, cx = H // 2, W // 2

        y_coords = torch.arange(H, device=fft_spectrum.device).float() - cy
        x_coords = torch.arange(W, device=fft_spectrum.device).float() - cx
        gy, gx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        dist_sq = gx ** 2 + gy ** 2

        accumulated = torch.zeros_like(fft_spectrum)
        for b in range(self.num_bands):
            sigma = self.base_sigma + b * self.sigma_step
            high_pass = 1.0 - torch.exp(-dist_sq / (2.0 * sigma ** 2))
            accumulated += high_pass * fft_spectrum

        acc_min = accumulated.min()
        acc_max = accumulated.max()
        if acc_max > acc_min:
            accumulated = (accumulated - acc_min) / (acc_max - acc_min)

        threshold = self._triangle_threshold(accumulated)
        return accumulated > threshold

    def _triangle_threshold(self, values):
        """
        Triangle algorithm: finds threshold maximizing perpendicular distance
        from histogram to the line connecting peak and farthest non-zero bin.

        Args:
            values: (H, W) float32 in [0, 1]
        Returns:
            scalar float threshold in [0, 1]
        """
        hist = torch.histc(values.flatten(), bins=self.triangle_bins, min=0.0, max=1.0)
        hist = hist / hist.sum()

        peak_bin = torch.argmax(hist).item()
        non_zero = torch.nonzero(hist).squeeze(-1)
        if len(non_zero) < 2:
            return 0.5

        farthest_bin = non_zero[-1].item() if non_zero[-1] - peak_bin > peak_bin - non_zero[0] else non_zero[0].item()
        if farthest_bin == peak_bin:
            return 0.5

        x1, y1 = float(peak_bin), float(hist[peak_bin])
        x2, y2 = float(farthest_bin), float(hist[farthest_bin])
        line_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if line_len < 1e-6:
            return 0.5

        max_dist = -1.0
        best_bin = peak_bin
        start = min(peak_bin, farthest_bin)
        end = max(peak_bin, farthest_bin)
        for i in range(start, end + 1):
            x0, y0 = float(i), float(hist[i])
            dist = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / line_len
            if dist > max_dist:
                max_dist = dist
                best_bin = i

        return best_bin / self.triangle_bins

    def compute_masks(self, rgb_image):
        """
        Full pipeline: RGB -> FFT masks + scale map.

        Args:
            rgb_image: (3, H, W) float32 in [0, 1], or None if disabled
        Returns:
            dict with:
                'opacity_mask':  (H, W) bool, True = high-freq
                'high_freq_mask': (H, W) bool, dense grid
                'low_freq_mask':  (H, W) bool, sparse grid
                'scale_map':      (H, W) float32, per-pixel scale
        """
        H, W = rgb_image.shape[1], rgb_image.shape[2]
        device = rgb_image.device

        if not self.enabled:
            return {
                'opacity_mask': torch.ones(H, W, dtype=torch.bool, device=device),
                'high_freq_mask': torch.ones(H, W, dtype=torch.bool, device=device),
                'low_freq_mask': torch.ones(H, W, dtype=torch.bool, device=device),
                'scale_map': torch.full((H, W), self.scale_max, device=device),
            }

        fft_spectrum = self.generate_frequency(rgb_image)
        opacity_mask = self.multiLayer_spectrogram(fft_spectrum)

        gy, gx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij',
        )
        high_freq_mask = (gy % self.high_freq_stride == 0) & (gx % self.high_freq_stride == 0)
        low_freq_mask = (gy % self.low_freq_stride == 0) & (gx % self.low_freq_stride == 0)

        fft_norm = (fft_spectrum - fft_spectrum.min()) / (fft_spectrum.max() - fft_spectrum.min() + 1e-8)
        scale_map = self.scale_max - (self.scale_max - self.scale_min) * fft_norm

        return {
            'opacity_mask': opacity_mask,
            'high_freq_mask': high_freq_mask,
            'low_freq_mask': low_freq_mask,
            'scale_map': scale_map,
        }
