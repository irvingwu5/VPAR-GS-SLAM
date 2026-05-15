# DPVO 位姿先验集成设计

## Context

VPAR-GS-SLAM 当前使用上一帧位姿作为当前帧 tracking 的初始值（`slam_frontend.py:129-130`），需要 ~100 轮渲染优化才能收敛。DPVO 作为单目深度视觉里程计，可以在 GPU 上快速估计帧间相对位姿（~5-15ms/帧），为 render-based tracking 提供更好的初始位姿，从而减少渲染迭代次数、提升 FPS。

核心约束：DPVO 是单目 VO（任意尺度），但 VPAR-GS-SLAM 的 RGB-D 模式使用物理尺度（米）。通过只取 Δ 位姿增量（相对变换）来规避尺度问题。

参考实现：FVO-GS-SLAM（`SimpleRGBDVO_Integration_Analysis.md`）的多候选仲裁 + VO 轨道模式。

## 设计决策

| 决策点 | 选择 | 依据 |
|--------|------|------|
| 进程模型 | DPVO 独立子进程 | GPU 显存隔离，通过 mp.Queue 通信 |
| 尺度处理 | 只取 Δ 位姿增量 | DPVO 帧间相对变换 × 上一帧 metric 位姿 |
| 候选策略 | 轻量仲裁（DPVO vs 恒速模型） | 用 DPVO flow magnitude 做质量门，无 render precheck |
| 初始化空白期 | 前 8 帧回退到上一帧位姿 | DPVO 需要 8 帧初始化后才输出有效位姿 |
| 迭代缩减 | 100 → 60（DPVO 采纳时） | 参考 FVO-GS-SLAM 的 40% 缩减 |

## 整体架构

```
Main Process (FrontEnd)                  DPVO Process (子进程)
       │                                        │
       │  dpvo_cmd_queue ────────────────────► │ ("track", frame_id, rgb_tensor, K_tensor)
       │                                        │
       │  dpvo_result_queue ◄────────────────── │ (frame_id, delta_c2w, flow_quality, ok)
       │                                        │
  tracking():                                   DPVO.__call__()
    ├─ _build_candidates()                        ├─ motion_probe() → quality
    │   - constant_velocity                       ├─ update() → optimize
    │   - dpvo_delta (if ready)                   └─ keyframe() → manage buffer
    ├─ _select_candidate()
    │   - dpvo flow gate → accept/reject
    ├─ tracking optimization
    │   60 iters (dpvo) / 100 iters (fallback)
    └─ update_pose()
```

## 消息协议

### dpvo_cmd_queue (FrontEnd → DPVO)

| 消息 | 数据 |
|------|------|
| `("track", frame_id, rgb_tensor (3,H,W) uint8 cuda, K_tensor (4,) float cuda)` | 逐帧追踪 |
| `("stop",)` | 终止进程 |

### dpvo_result_queue (DPVO → FrontEnd)

| 消息 | 数据 |
|------|------|
| 成功 | `(frame_id, delta_c2w (4x4 np), flow_quality (float), True)` |
| 失败/初始化中 | `(frame_id, None, 0.0, False)` |

## 模块设计

### 1. DPVOProvider (`utils/dpvo_prior/__init__.py`)

新增桥接类，封装 DPVO 子进程生命周期和位姿提取。

```
class DPVOProvider:
    __init__(config, W, H, fx, fy, cx, cy)
        → 加载 DPVO 配置 + 权重 → 创建 DPVO 子进程 → 启动通信线程

    track(rgb_uint8_np, prev_metric_c2w) → (success, est_c2w, info)
        1. 将 numpy image 搬上 GPU
        2. 发送 ("track", fid, image_cuda, K_cuda) 到 cmd_queue
        3. 从 result_queue 收到 delta_c2w
        4. est_c2w = prev_metric_c2w @ delta_c2w  (Δ 累积)
        5. 质量门：flow_quality > flow_quality_thresh → success=True
        返回 (success, est_c2w, {flow_quality, runtime_ms})

    reset()
        → stop() + 重建 DPVO 子进程

    is_initialized() → bool
        DPVO 内部 n >= 8 且 is_initialized == True

    stop()
        → 发送 ("stop",) → join 子进程
```

**DPVO 子进程主循环**（运行在独立 mp.Process 中）：
```
while True:
    msg = cmd_queue.get()
    if msg[0] == "stop": save trajectory, break
    if msg[0] == "track":
        fid, image, K = msg
        if slam is None: slam = DPVO(cfg, weight, ht, wd)
        slam(tstamp=fid, image=image, intrinsics=K)
        if slam.is_initialized:
            delta_c2w = compute_delta(slam.pg.poses_[slam.n-1], slam.pg.poses_[slam.n-2])
            quality = slam.motion_probe()
            result_queue.put((fid, delta_c2w, quality, True))
        else:
            result_queue.put((fid, None, 0.0, False))
```

**compute_delta**：
```python
def compute_delta(cur_pose_vec, prev_pose_vec):
    """cur/prev 都是 [tx,ty,tz,qx,qy,qz,qw] W2C (DPVO 内部 convention)"""
    P_cur = SE3(cur_pose_vec)
    P_prev = SE3(prev_pose_vec)
    return (P_prev.inv() * P_cur).matrix().cpu().numpy()  # C2W delta
```

### 2. FrontEnd 修改 (`utils/slam_frontend.py`)

**`__init__` 新增属性：**
- `self.vo_prior_enabled` — 从 config 读取
- `self.vo_prior` — DPVOProvider 实例（懒加载）
- `self.tracking_itr_accepted` / `self.tracking_itr_fallback`
- `self.dpvo_warmup_frames` / `self.flow_quality_thresh`
- `self.last_metric_c2w` — 上一帧的 metric C2W（用于 Δ 累乘）

**新增 `_init_dpvo_prior()`：**
```python
def _init_dpvo_prior(self):
    if self.vo_prior is not None: return
    w, h = self.dataset.width, self.dataset.height
    fx, fy = self.dataset.fx, self.dataset.fy
    cx, cy = self.dataset.cx, self.dataset.cy
    self.vo_prior = DPVOProvider(self.config, w, h, fx, fy, cx, cy)
```

**新增 `_build_candidates(prev_c2w, dpvo_c2w)`：**
```python
def _build_candidates(self, prev_c2w, dpvo_c2w):
    candidates = {}
    # 恒速模型
    if len(self.last_two_c2w) >= 2:
        c2w_t1, c2w_t2 = self.last_two_c2w[-2], self.last_two_c2w[-1]
        cv_c2w = c2w_t1 @ np.linalg.inv(c2w_t2) @ c2w_t1  # 简化恒速
        candidates["constant_velocity"] = cv_c2w
    # DPVO Δ
    if dpvo_c2w is not None:
        candidates["dpvo_delta"] = dpvo_c2w
    return candidates
```

**修改 `tracking()` (line 128-130)：**
```python
# 原代码：
# prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
# viewpoint.update_RT(prev.R, prev.T)

# 新代码：
if self.vo_prior_enabled and cur_frame_idx > self.dpvo_warmup_frames:
    self._init_dpvo_prior()
    prev_cam = self.cameras[cur_frame_idx - 1]
    prev_c2w = np.linalg.inv(rt2mat(prev_cam.R, prev_cam.T).cpu().numpy())
    rgb_np = self._camera_rgb(viewpoint)
    dpvo_ok, dpvo_c2w, dpvo_info = self.vo_prior.track(rgb_np, prev_c2w)
    candidates = self._build_candidates(prev_c2w, dpvo_c2w if dpvo_ok else None)
    best_name, best_c2w = self._select_candidate(candidates, dpvo_info)
    use_dpvo = (best_name == "dpvo_delta")
else:
    use_dpvo = False
    prev_cam = self.cameras[cur_frame_idx - self.use_every_n_frames]
    best_c2w = np.linalg.inv(rt2mat(prev_cam.R, prev_cam.T).cpu().numpy())

# 应用初值
w2c = np.linalg.inv(best_c2w)
viewpoint.update_RT(
    torch.from_numpy(w2c[:3, :3]).float().cuda(),
    torch.from_numpy(w2c[:3, 3]).float().cuda(),
)

# 动态迭代次数
itr_num = self.tracking_itr_accepted if use_dpvo else self.tracking_itr_fallback
for tracking_itr in range(itr_num):
    ...
```

**新增辅助方法 `_camera_rgb()`：**
```python
def _camera_rgb(self, viewpoint):
    """Camera tensor (3,H,W) float [0,1] → numpy (H,W,3) uint8 [0,255]"""
    img = viewpoint.original_image.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)
```

**`_select_candidate(candidates, dpvo_info)`：**
```python
def _select_candidate(self, candidates, dpvo_info):
    # DPVO 优先：flow quality 足够好就用 DPVO
    if "dpvo_delta" in candidates:
        flow_qual = dpvo_info.get("flow_quality", 0.0)
        if flow_qual > self.flow_quality_thresh:
            return "dpvo_delta", candidates["dpvo_delta"]
    # Fallback: constant_velocity
    if "constant_velocity" in candidates:
        return "constant_velocity", candidates["constant_velocity"]
    # Last resort: previous frame (由调用者提供)
    return "previous", None
```

**submap 重置时调用 `self.vo_prior.reset()`.**

### 3. 配置新增

在相关 `base_config.yaml` 中新增：

```yaml
VOPrior:
  enabled: True
  type: "dpvo"
  warmup_frames: 10
  tracking_itr_accepted: 60
  tracking_itr_fallback: 100
  flow_quality_thresh: 1.5
  dpvo_queue_timeout: 0.05

DPVO:
  weight_path: "weights/dpvo.pth"
  config_type: "default"
  patches_per_frame: 96
  optimization_window: 10
  patch_lifetime: 13
```

### 4. 姿态约定

| 层 | 内部表示 | 说明 |
|-----|---------|------|
| VPAR-GS-SLAM Camera | W2C (4x4) | `camera.R, camera.T` |
| DPVO 内部 (`pg.poses_`) | W2C (7-vec) | `[tx,ty,tz,qx,qy,qz,qw]`，qw 最后 |
| DPVO `terminate()` 输出 | C2W (N,7) | 输出前 `.inv()` |
| DPVOProvider `track()` 返回 | C2W (4x4 np) | delta: `P_prev^{-1} * P_cur` |

转换路径：
```
DPVO 内部:  P_wc (7-vec)  →  SE3(P_cur) * SE3(P_prev).inv()  →  4x4 C2W delta  →  prev_metric_c2w @ delta  →  W2C via np.linalg.inv()  →  camera.update_RT()
```

## 实施阶段

### 阶段 1：DPVO 环境就绪
- 编译 CUDA 扩展（`python setup.py build_ext --inplace`）
- 获取 DPVO 模型权重文件
- 验证可导入并单帧推理

### 阶段 2：DPVOProvider 桥接层
- 创建 `utils/dpvo_prior/__init__.py`
- DPVO 子进程 + 双队列通信 + Δ 位姿计算 + 质量门
- 独立测试

### 阶段 3：FrontEnd 集成
- 修改 `slam_frontend.py`：初始化、candidate 构建/选择、tracking 注入
- 处理 warmup、submap 重置

### 阶段 4：配置 + 端到端验证
- 添加配置节
- 跑完整序列，对比 ATE / FPS / 渲染质量

## 验证

- [ ] `python -c "from dpvo.dpvo import DPVO"` 导入成功
- [ ] DPVO 子进程可独立处理一帧并返回位姿
- [ ] `DPVOProvider.track()` 返回正确的 C2W 位姿
- [ ] VPAR-GS-SLAM 在 office0 上正常追踪，ATE 不低于 baseline
- [ ] DPVO 采纳帧的 tracking 迭代次数 = 60
- [ ] FPS 对比 baseline 有提升
- [ ] 无 GPU 显存泄漏
