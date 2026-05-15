"""DPVO Pose Prior Bridge — wraps DPVO monocular VO in a subprocess for VPAR-GS-SLAM."""

import os
import torch
import torch.multiprocessing as mp
import numpy as np


def _fmt_mb(t):
    """Format tensor size in MB, return 'N/A' if None or not a tensor."""
    if t is None or not isinstance(t, torch.Tensor):
        return "N/A"
    return f"{t.element_size() * t.numel() / 1024 / 1024:.2f} MB"


def _log_gpu_memory(slam, frame_id):
    """Print per-frame GPU memory breakdown of all major DPVO buffers."""
    pg = slam.pg
    alloc = torch.cuda.memory_allocated() / 1024 / 1024
    rsvd = torch.cuda.memory_reserved() / 1024 / 1024

    n_active = pg.ii.numel()
    n_inactive = pg.ii_inac.numel()
    net_mb = _fmt_mb(pg.net)
    target_mb = _fmt_mb(getattr(pg, 'target', None))
    target_inac_mb = _fmt_mb(getattr(pg, 'target_inac', None))

    print(f"[DPVO MEM] frame {frame_id} | n={slam.n} m={slam.m} | "
          f"alloc={alloc:.1f} MB reserved={rsvd:.1f} MB",
          flush=True)
    print(f"  edges: active={n_active} inactive={n_inactive} "
          f"net={net_mb} target={target_mb} target_inac={target_inac_mb}",
          flush=True)
    print(f"  fmap1={_fmt_mb(slam.fmap1_)} fmap2={_fmt_mb(slam.fmap2_)} "
          f"imap={_fmt_mb(slam.imap_)} gmap={_fmt_mb(slam.gmap_)}",
          flush=True)
    print(f"  patches={_fmt_mb(pg.patches_)} net_shape={list(pg.net.shape)} "
          f"ii_inac_shape={list(pg.ii_inac.shape)}",
          flush=True)


def _dpvo_process(cfg, weight_path, ht, wd, device_id, cmd_queue, result_queue):
    """Run DPVO in a dedicated subprocess (spawn context, own CUDA stream)."""
    torch.cuda.set_device(device_id)

    from dpvo.dpvo import DPVO
    from dpvo.lietorch import SE3

    slam = None
    try:
        while True:
            msg = cmd_queue.get()
            if msg[0] == "stop":
                if slam is not None:
                    slam.terminate()
                break
            elif msg[0] == "track":
                frame_id, image_cpu, K_cpu = msg[1:]
                image = image_cpu.cuda(device_id)
                intrinsics = K_cpu.cuda(device_id)
                if slam is None:
                    slam = DPVO(cfg, weight_path, ht=ht, wd=wd, viz=False)

                with torch.no_grad():
                    slam(tstamp=frame_id, image=image, intrinsics=intrinsics)

                # Memory profiling disabled after OOM fix confirmed. Keep helpers for debugging.
                # if frame_id % 5 == 0 and slam is not None:
                #     _log_gpu_memory(slam, frame_id)

                if slam.is_initialized:
                    n = slam.n
                    P_prev = SE3(slam.pg.poses_[n - 2 : n - 1])
                    P_cur = SE3(slam.pg.poses_[n - 1 : n])
                    # Internal poses are W2C. delta = C2W_{t-1}^{-1} @ C2W_t = P_{t-1} * P_t^{-1}
                    delta_c2w = (P_prev * P_cur.inv()).matrix()[0].cpu().numpy()
                    result_queue.put((frame_id, delta_c2w, 0.0, True))
                else:
                    result_queue.put((frame_id, None, 0.0, False))
            else:
                raise ValueError(f"Unknown DPVO command: {msg[0]}")
    except Exception:
        result_queue.put((-1, None, 0.0, False))
        import traceback
        traceback.print_exc()


class DPVOProvider:
    """Manages DPVO subprocess lifecycle. Provides per-frame relative C2W pose deltas."""

    def __init__(self, config, W, H, fx, fy, cx, cy):
        self.config = config
        self.W = W
        self.H = H
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self._frame_counter = 0
        self._n_initialized_frames = 0
        self.process = None

        # GPU device for DPVO (default: share with main process)
        self._dpvo_device = config["VOPrior"].get("dpvo_device_id", 0)

        # Build DPVO YACS config
        from dpvo.config import cfg as _dpvo_cfg
        config_file = config["DPVO"]["config_file"]
        if not os.path.isabs(config_file):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_file = os.path.join(repo_root, config_file)
        _dpvo_cfg.merge_from_file(config_file)
        for key in ["PATCHES_PER_FRAME", "OPTIMIZATION_WINDOW", "PATCH_LIFETIME",
                     "REMOVAL_WINDOW", "KEYFRAME_THRESH", "BUFFER_SIZE"]:
            if key in config["DPVO"]:
                setattr(_dpvo_cfg, key, config["DPVO"][key])

        self.dpvo_cfg = _dpvo_cfg.clone()

        weight_path = config["DPVO"]["weight_path"]
        if not os.path.isabs(weight_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            weight_path = os.path.join(repo_root, weight_path)
        self.weight_path = weight_path

        self._start_process()

    def _start_process(self):
        ctx = mp.get_context("spawn")
        self.cmd_queue = ctx.Queue(maxsize=1)
        self.result_queue = ctx.Queue(maxsize=1)
        self.process = ctx.Process(
            target=_dpvo_process,
            args=(self.dpvo_cfg, self.weight_path, self.H, self.W,
                  self._dpvo_device, self.cmd_queue, self.result_queue),
            daemon=True,
        )
        self.process.start()

    def track(self, rgb_uint8_np, prev_metric_c2w):
        """
        Args:
            rgb_uint8_np:  (H, W, 3) numpy uint8 [0, 255]
            prev_metric_c2w: (4, 4) numpy float64, previous frame's metric C2W matrix
        Returns:
            success: bool
            est_c2w: (4, 4) numpy float64, estimated current C2W (metric scale)
            info:    dict with keys "flow_quality", optionally "error"
        """
        # Send CPU tensors to avoid CUDA IPC memory accumulation
        image_cpu = torch.from_numpy(rgb_uint8_np.copy()).permute(2, 0, 1)  # (3, H, W) uint8 CPU
        K_cpu = torch.tensor([self.fx, self.fy, self.cx, self.cy],
                             dtype=torch.float32)  # (4,) float32 CPU

        self.cmd_queue.put(("track", self._frame_counter, image_cpu, K_cpu))
        self._frame_counter += 1

        timeout = self.config["VOPrior"].get("dpvo_queue_timeout", 0.5)
        try:
            fid, delta_c2w, quality, ok = self.result_queue.get(timeout=timeout)
        except Exception:
            return False, prev_metric_c2w, {"error": "timeout"}

        if not ok or delta_c2w is None:
            return False, prev_metric_c2w, {"flow_quality": quality}

        self._n_initialized_frames += 1

        # Anti-drift: apply DPVO delta to previous metric pose
        est_c2w = prev_metric_c2w @ delta_c2w
        info = {"flow_quality": quality}
        return True, est_c2w, info

    def is_initialized(self):
        return self._n_initialized_frames > 0

    def reset(self):
        self.stop()
        self._frame_counter = 0
        self._n_initialized_frames = 0
        self._start_process()

    def stop(self):
        try:
            self.cmd_queue.put(("stop",))
        except Exception:
            pass
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
            self.process = None

    def __del__(self):
        self.stop()
