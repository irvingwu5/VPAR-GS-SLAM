import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.error_mask import ErrorMaskDensifier
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        self.fft_masks = {}
        self.error_mask_densifier = None

        # ===== PAR-RSKM =====
        par_cfg = config.get("PAR_RSKM", {})
        self.use_par_rskm = par_cfg.get("enabled", False)
        self.par_config = {}
        self._par_reliabilities = {}
        self._par_rskm_log_counter = 0
        self._par_rskm_stats = None
        if self.use_par_rskm:
            self.par_config = {
                "beta_pose": float(par_cfg.get("beta_pose", 1.0)),
                "eps": float(par_cfg.get("eps", 1.0e-6)),
                "gamma": float(par_cfg.get("gamma", 1.5)),
                "tau_r": float(par_cfg.get("reliability_threshold", 0.05)),
                "w_min": float(par_cfg.get("min_weight", 0.25)),
                "w_max": float(par_cfg.get("max_weight", 1.0)),
                "default_reliability": float(par_cfg.get("default_reliability", 1.0)),
                "log_interval": int(par_cfg.get("log_interval", 50)),
            }
            self._par_rskm_debug_log = par_cfg.get("debug_log", False)
            self._par_use_temporal_bins = par_cfg.get("use_temporal_bins", False)
            self._par_bin_probs = {
                "recent": float(par_cfg.get("recent_bin_prob", 0.5)),
                "middle": float(par_cfg.get("middle_bin_prob", 0.3)),
                "old": float(par_cfg.get("old_bin_prob", 0.2)),
            }
            rskm_seed = par_cfg.get("seed", config.get("Experiment", {}).get("seed", 42))
            self.par_rskm_rng = random.Random(rskm_seed)
            Log(f"[PAR-RSKM] enabled beta={self.par_config['beta_pose']} "
                f"gamma={self.par_config['gamma']} tau_r={self.par_config['tau_r']} "
                f"temporal_bins={self._par_use_temporal_bins}")

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.use_normal = self.config["opt_params"].get("lambda_normal", 0.0) > 0
        self.normal_apply_iters = self.config["opt_params"].get("normal_apply_iters", 1)
        self.normal_start = self.config["opt_params"].get("normal_start_iter", 0)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.fft_masks = {}
        self._par_reliabilities = {}
        self._par_rskm_stats = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    errormask_enabled = self.config.get("errormask", {}).get(
                        "enabled", False
                    )
                    prune_scale_th = self.config.get("errormask", {}).get(
                        "prune_scale_th", 10.0
                    )
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                        use_fgs_pruning=errormask_enabled,
                        prune_scale_th=prune_scale_th,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        current_kf_id = current_window[-1] if len(current_window) > 0 else None

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            # ---- 选择监督帧 ----
            if self.use_par_rskm and not prune:
                num_samples = len(current_window) + 2
                supervised_kf_ids = self._select_par_rskm_keyframes(current_window, num_samples)
                supervision_pairs = [(kf_id, self.viewpoints[kf_id]) for kf_id in supervised_kf_ids]
                keyframes_opt = viewpoint_stack[:]
            else:
                supervision_pairs = [(kf_idx, self.viewpoints[kf_idx]) for kf_idx in current_window]
                for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                    supervision_pairs.append((None, random_viewpoint_stack[cam_idx]))
                keyframes_opt = viewpoint_stack[:]

            apply_normal = _ >= self.normal_start and (
                self.normal_apply_iters == 0 or _ < self.normal_start + self.normal_apply_iters
            )

            for kf_idx, viewpoint in supervision_pairs:
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                rend_dist = render_pkg["rend_dist"]

                loss_view = get_loss_mapping(
                    self.config,
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    rend_normal=render_pkg.get("rend_normal", None),
                    rend_dist=rend_dist,
                    gt_normal_cam=viewpoint.normal,
                    gt_normal_mask=viewpoint.normal_mask,
                    apply_normal=apply_normal,
                )

                # ---- PAR-RSKM: apply replay loss weight ----
                if (self.use_par_rskm
                        and kf_idx is not None
                        and kf_idx != current_kf_id):
                    from utils.par_rskm import compute_replay_weight
                    r = getattr(viewpoint, "par_reliability", None)
                    w = compute_replay_weight(
                        r,
                        min_weight=self.par_config.get("w_min", 0.25),
                        max_weight=self.par_config.get("w_max", 1.0),
                    )
                    loss_view = w * loss_view

                loss_view.backward()

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append((kf_idx, n_touched))

                del render_pkg
                torch.cuda.empty_cache()

            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for kf_idx, n_touched in n_touched_acm:
                    if kf_idx is not None and kf_idx in current_window_set:
                        self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    errormask_enabled = self.config.get("errormask", {}).get(
                        "enabled", False
                    )

                    if errormask_enabled and self.error_mask_densifier is None:
                        self.error_mask_densifier = ErrorMaskDensifier(
                            self.config, self.gaussians
                        )

                    if errormask_enabled and self.error_mask_densifier is not None:
                        for idx in range(len(current_window)):
                            kf_idx = current_window[idx]
                            vp = viewpoint_stack[idx]
                            kf_fft_masks = self.fft_masks.get(kf_idx)
                            render_pkg = render(
                                vp, self.gaussians, self.pipeline_params,
                                self.background,
                            )
                            error_mask, _ = self.error_mask_densifier.compute_error_mask(
                                render_pkg, vp
                            )
                            pixel_indices = self.error_mask_densifier.select_pixels(
                                error_mask, kf_fft_masks
                            )
                            self.error_mask_densifier.create_gaussians_at_pixels(
                                vp, pixel_indices, kf_fft_masks, kf_idx,
                            )

                        prune_scale_th = self.config.get("errormask", {}).get(
                            "prune_scale_th", 10.0
                        )
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                            use_fgs_pruning=True,
                            prune_scale_th=prune_scale_th,
                            skip_clone_split=True,
                        )
                    else:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                        )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)

        # ---- PAR-RSKM stats summary ----
        if (self.use_par_rskm
                and self._par_rskm_stats is not None
                and self._par_rskm_stats["reliability_count"] > 0):
            st = self._par_rskm_stats
            mean_r = st["reliability_sum"] / st["reliability_count"]
            mean_w = st["weight_sum"] / st["weight_count"] if st["weight_count"] > 0 else 1.0
            if self._par_rskm_debug_log:
                Log(f"[PAR-RSKM][Stats] current={st['num_current_selected']} "
                    f"replay={st['num_replay_selected']} "
                    f"fallback={st['num_replay_fallback']} "
                    f"rejected={st['num_replay_rejected']} "
                    f"mean_r={mean_r:.3f} mean_w={mean_w:.3f} "
                    f"pool={len(self.viewpoints)}kfs")
            self._par_rskm_stats = None

        return gaussian_split

    # ========================================================================
    # ========================================================================
    # 6.1 RSKM (Random Sampling Keyframe Mapping) — vanilla fallback
    # ========================================================================
    def _select_rskm_keyframes(self, current_window, num_samples):
        active_kf_ids = sorted(list(self.viewpoints.keys()))
        current_kf_id = current_window[-1] if len(current_window) > 0 else None

        selected = []
        for s in range(num_samples):
            iter_id = self.iteration_count + s
            if iter_id % self.config.get("PAR_RSKM", {}).get("current_frame_interval", 4) == 0:
                if current_kf_id is not None and current_kf_id in self.viewpoints:
                    selected.append(current_kf_id)
                    continue
            if len(active_kf_ids) <= 1 and current_kf_id is not None:
                selected.append(current_kf_id)
            elif len(active_kf_ids) > 0:
                selected.append(self.par_rskm_rng.choice(active_kf_ids))
            elif current_kf_id is not None:
                selected.append(current_kf_id)
        return selected

    # ========================================================================
    # 6.2 PAR-RSKM: Pose-Aware Random Keyframe Replay
    # ========================================================================
    def _select_par_rskm_keyframes(self, current_window, num_samples):
        from utils.par_rskm import (
            compute_reliabilities_batch, compute_par_sampling_score,
            select_par_keyframes, select_par_keyframes_binned,
        )

        current_kf_id = current_window[-1] if len(current_window) > 0 else None

        # Step 1: Update reliabilities for newly initialized viewpoints
        newly_initialized = any(
            getattr(vp, "par_initialized", False)
            and getattr(vp, "par_reliability", None) is None
            for vp in self.viewpoints.values()
        )
        if newly_initialized:
            self._par_reliabilities = compute_reliabilities_batch(
                self.viewpoints, self.par_config
            )
            if self._par_rskm_debug_log:
                rel_vals = list(self._par_reliabilities.values())
                if rel_vals:
                    Log(f"[PAR-RSKM] reliabilities: n={len(rel_vals)} "
                        f"mean={sum(rel_vals)/len(rel_vals):.4f} "
                        f"min={min(rel_vals):.4f} max={max(rel_vals):.4f}")

        reliabilities = self._par_reliabilities
        if not reliabilities:
            if self._par_rskm_debug_log:
                Log(f"[PAR-RSKM] no reliabilities yet (pool={len(self.viewpoints)}), "
                    f"fallback to vanilla uniform")
            return self._select_rskm_keyframes(current_window, num_samples)

        # Step 2: Compute sampling scores
        scores = {}
        active_kf_ids = sorted(list(self.viewpoints.keys()))
        for kf_id in active_kf_ids:
            vp = self.viewpoints[kf_id]
            r = reliabilities.get(kf_id, self.par_config["default_reliability"])
            replay_cnt = getattr(vp, "par_replay_count", 0)
            scores[kf_id] = compute_par_sampling_score(
                r, replay_count=replay_cnt,
                threshold=self.par_config["tau_r"],
                gamma=self.par_config["gamma"],
                eps=self.par_config["eps"],
            )

        # Step 3: Weighted selection
        interval = self.config.get("PAR_RSKM", {}).get("current_frame_interval", 4)
        if self._par_use_temporal_bins:
            selected, fallback_used = select_par_keyframes_binned(
                self.viewpoints, current_window, num_samples,
                scores, self._par_bin_probs, self.par_rskm_rng,
                interval, self.iteration_count,
            )
        else:
            selected, fallback_used = select_par_keyframes(
                self.viewpoints, current_window, num_samples,
                scores, self.par_rskm_rng,
                interval, self.iteration_count,
            )

        # Step 4: Update replay counts
        for kf_id in selected:
            if kf_id != current_kf_id and kf_id in self.viewpoints:
                vp = self.viewpoints[kf_id]
                vp.par_replay_count = getattr(vp, "par_replay_count", 0) + 1
                vp.par_last_replay_iter = self.iteration_count

        # Step 5: Accumulate stats
        self._par_rskm_log_counter += 1
        if self._par_rskm_stats is None:
            self._par_rskm_stats = {
                "num_current_selected": 0, "num_replay_selected": 0,
                "num_replay_fallback": 0, "num_replay_rejected": 0,
                "reliability_sum": 0.0, "reliability_count": 0,
                "weight_sum": 0.0, "weight_count": 0,
            }

        for kf_id in selected:
            if kf_id == current_kf_id:
                self._par_rskm_stats["num_current_selected"] += 1
            else:
                self._par_rskm_stats["num_replay_selected"] += 1
                r = reliabilities.get(kf_id, self.par_config["default_reliability"])
                self._par_rskm_stats["reliability_sum"] += r
                self._par_rskm_stats["reliability_count"] += 1
                w = max(self.par_config["w_min"], min(self.par_config["w_max"], r))
                self._par_rskm_stats["weight_sum"] += w
                self._par_rskm_stats["weight_count"] += 1
        if fallback_used:
            self._par_rskm_stats["num_replay_fallback"] += 1

        tau_r = self.par_config["tau_r"]
        for kf_id in active_kf_ids:
            r = reliabilities.get(kf_id, self.par_config["default_reliability"])
            if r < tau_r:
                self._par_rskm_stats["num_replay_rejected"] += 1

        # Periodic log
        if self._par_rskm_debug_log and self._par_rskm_log_counter % self.par_config.get("log_interval", 50) == 0:
            rel_values = [v for v in reliabilities.values() if v is not None]
            mean_r = sum(rel_values) / max(len(rel_values), 1) if rel_values else 0
            sample_info = []
            for kf_id in selected[:3]:
                r = reliabilities.get(kf_id, 0)
                s = scores.get(kf_id, 0)
                rc = getattr(self.viewpoints.get(kf_id), "par_replay_count", 0)
                sample_info.append(f"kf={kf_id} r={r:.3f} s={s:.4f} rc={rc}")
            Log(f"[PAR-RSKM] candidates={len(active_kf_ids)} "
                f"selected={len(selected)} mean_r={mean_r:.3f} "
                + " | ".join(sample_info))

        return selected

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    fft_masks = data[5] if len(data) > 5 else None

                    if fft_masks is not None:
                        self.fft_masks[cur_frame_idx] = {
                            k: v.cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in fft_masks.items()
                        }

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

                    if len(self.fft_masks) > self.window_size * 2:
                        keep_ids = set(current_window)
                        self.fft_masks = {
                            k: v for k, v in self.fft_masks.items() if k in keep_ids
                        }

                    # ---- PAR-RSKM: compute reliabilities for new keyframe ----
                    if self.use_par_rskm and len(self.viewpoints) >= 2:
                        from utils.par_rskm import compute_reliabilities_batch
                        self._par_reliabilities = compute_reliabilities_batch(
                            self.viewpoints, self.par_config
                        )

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
