import os
import subprocess
import sys
import threading
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class GPUMemoryMonitor:
    """Background thread that tracks peak physical GPU memory across all processes.

    Uses pynvml if available (more reliable), falls back to nvidia-smi CLI.
    Polls every 2s; subtracts baseline (measured before SLAM starts) to report
    net usage excluding pre-existing allocations.
    """

    def __init__(self, physical_gpu_id=0):
        self.keep_measuring = True
        self.peak_memory = 0
        self.baseline_memory = 0
        self.physical_gpu_id = physical_gpu_id
        self._backend = None  # "pynvml" or "nvidia-smi" or None
        self._nvml_handle = None
        self.thread = threading.Thread(target=self._measure_usage, daemon=True)

        if self._try_init_pynvml():
            self._backend = "pynvml"
            Log(f"[GPU Mem] Using pynvml backend (GPU {physical_gpu_id})")
        elif self._check_nvidia_smi():
            self._backend = "nvidia-smi"
            Log(f"[GPU Mem] Using nvidia-smi backend (GPU {physical_gpu_id})")
        else:
            self._backend = None
            Log("[GPU Mem] WARNING: No GPU monitoring backend available. "
                "Install pynvml or ensure nvidia-smi is on PATH.")

    def _try_init_pynvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            self._pynvml_available = True
            return True
        except Exception:
            self._pynvml_available = False
            return False

    def _check_nvidia_smi(self):
        try:
            subprocess.check_output(
                ['nvidia-smi', f'--id={self.physical_gpu_id}',
                 '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8', stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def _query_gpu_mem_pynvml(self):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used // (1024 * 1024)  # bytes -> MB

    def _query_gpu_mem_nvsmi(self):
        result = subprocess.check_output(
            ['nvidia-smi', f'--id={self.physical_gpu_id}',
             '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        return int(result.strip())

    def _measure_usage(self):
        query = (self._query_gpu_mem_pynvml if self._backend == "pynvml"
                 else self._query_gpu_mem_nvsmi)
        while self.keep_measuring:
            try:
                current_mem = query()
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except Exception as e:
                Log(f"[GPU Mem] Poll error: {e}")
            time.sleep(2.0)

    def start(self):
        if self._backend is None:
            return
        query = (self._query_gpu_mem_pynvml if self._backend == "pynvml"
                 else self._query_gpu_mem_nvsmi)
        try:
            self.baseline_memory = query()
            Log(f"[GPU Mem] Baseline: {self.baseline_memory} MB")
        except Exception as e:
            self.baseline_memory = 0
            Log(f"[GPU Mem] Baseline query failed: {e}")
        self.thread.start()

    def stop(self):
        self.keep_measuring = False
        time.sleep(0.2)
        net = max(0, self.peak_memory - self.baseline_memory)
        Log(f"[GPU Mem] Peak: {self.peak_memory} MB | "
            f"Baseline: {self.baseline_memory} MB | "
            f"Net (peak-baseline): {net} MB")
        return net


class SLAM:
    def __init__(self, config, save_dir=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        # Merge surface_depth config block into pipeline_params for unified access
        if "surface_depth" in config:
            config["pipeline_params"]["surface_depth"] = config["surface_depth"]
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        # ---- DPVO Pose Prior ----
        vop = config.get("VOPrior", {})
        self.vo_prior = None
        if vop.get("enabled", False) and vop.get("type", "none") == "dpvo":
            from utils.dpvo_prior import DPVOProvider
            self.vo_prior = DPVOProvider(
                config,
                W=self.dataset.width,
                H=self.dataset.height,
                fx=self.dataset.fx,
                fy=self.dataset.fy,
                cx=self.dataset.cx,
                cy=self.dataset.cy,
            )
            Log(f"[VOPrior] DPVO provider initialized by slam.py "
                f"(W={self.dataset.width}, H={self.dataset.height})")
        self.frontend.set_vo_prior(self.vo_prior)

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        # Shutdown DPVO prior
        if self.vo_prior is not None:
            self.vo_prior.stop()
            Log("[VOPrior] DPVO process stopped by slam.py")

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        # Use frontend gaussians (updated by backend mapping), not the init-time empty shell
        self.gaussians = self.frontend.gaussians
        num_gaussians = self.gaussians.get_xyz.shape[0]
        Log(f"Number of Gaussians: {num_gaussians}", tag="Eval")

        if self.eval_rendering:
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
                save_rgb=False,
                save_depth=False,
                save_normal=False,
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "sync_backend" and frontend_queue.empty():
                    gaussians = data[1]
                    self.gaussians = gaussians
                    break

            render_cfg = self.config.get("Results", {})
            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="after_opt",
                save_rgb=render_cfg.get("save_render_rgb", False),
                save_depth=render_cfg.get("save_render_depth", False),
                save_normal=render_cfg.get("save_render_normal", False),
            )
            metrics_table.add_data(
                "After",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

            ply_path = os.path.join(
                self.save_dir, "point_cloud", "final", "point_cloud.ply"
            )
            if os.path.exists(ply_path):
                map_size_mb = os.path.getsize(ply_path) / (1024 * 1024)
                Log(f"Final Map Size (PLY): {map_size_mb:.2f} MB", tag="Eval")

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running VPAR-GS-SLAM in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=False")
        config["Results"]["use_wandb"] = False

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="vpar-gs-slam",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    # GPU Memory Monitor (nvidia-smi physical peak, paper metric)
    gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    physical_gpu_id = int(gpu_id_str.split(',')[0])
    mem_monitor = GPUMemoryMonitor(physical_gpu_id=physical_gpu_id)
    mem_monitor.start()
    Log(f"Started tracking physical GPU {physical_gpu_id} memory...")

    slam = SLAM(config, save_dir=save_dir)
    slam.run()

    if save_dir is not None:
        real_peak_memory_mb = mem_monitor.stop()
        Log(f"System Physical Peak GPU Memory (nvidia-smi): {real_peak_memory_mb:.2f} MB", tag="Eval")

    wandb.finish()

    # All done
    Log("Done.")
