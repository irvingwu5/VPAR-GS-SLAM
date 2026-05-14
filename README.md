# VPAR-GS-SLAM: Visual Odometry Prior Guided Tracking and Pose-Aware Replay for RGB-D 2D Gaussian SLAM

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          slam.py (主进程)                            │
│  解析配置 → 加载数据集 → 创建 GaussianModel → 启动后端/可视化进程       │
│                         → 运行前端主循环                              │
└───────────────┬─────────────────────────┬───────────────────────────┘
                │                         │
   ┌────────────▼──────────┐   ┌─────────▼──────────┐
   │   FrontEnd (主进程)    │   │  BackEnd (子进程)    │
   │   utils/slam_frontend  │   │  utils/slam_backend │
   │                        │   │                     │
   │  • 相机位姿追踪         │   │  • 高斯地图优化       │
   │  • 关键帧选取           │   │  • 高斯增删/修剪      │
   │  • 损失计算             │   │  • 关键帧位姿优化(BA) │
   │  • GUI通信              │   │  • 颜色精炼           │
   └───────┬────────────────┘   └──────────┬──────────┘
           │ backend_queue.put()          │ frontend_queue.put()
           │  "init"/"keyframe"           │  "sync_backend"
           └──────────────────────────────┘
                        进程间通信 (mp.Queue)

   ┌──────────────────────────┐
   │  GUI (子进程, 可选)       │
   │  gui/slam_gui.py         │
   │  • OpenGL 实时渲染        │
   │  • 交互控制 (暂停/恢复)    │
   └──────┬───────────────────┘
          │ q_main2vis / q_vis2main
          └────── 与主进程通信
```

## 模块说明

### 1. 主入口 — `slam.py`

**类 `SLAM`**：系统编排器，负责：

- 加载 YAML 配置（`configs/`），解析为 `model_params` / `opt_params` / `pipeline_params`
- 创建 `GaussianModel`（2D 高斯场景表示）
- 加载数据集（`utils/dataset.py`）
- 创建 3 个进程间通信队列：`frontend_queue`、`backend_queue`、`q_main2vis`/`q_vis2main`
- 初始化 `FrontEnd` / `BackEnd` / `ParamsGUI`，注入共享对象
- 依次启动 GUI 进程 → Backend 进程 → 运行 Frontend 主循环
- 运行结束后执行：ATE 轨迹评估 → 颜色精炼 → 渲染评估 (PSNR/SSIM/LPIPS) → 保存高斯模型

### 2. 前端追踪 — `utils/slam_frontend.py`

**类 `FrontEnd(mp.Process)`**：运行在主进程中，负责实时相机追踪与关键帧决策。

核心方法：

| 方法 | 功能 |
|------|------|
| `run()` | 主循环：逐帧读取数据集，执行初始化 → 追踪 → 关键帧选取 |
| `initialize()` | 首帧初始化：以真值位姿设定首帧，添加关键帧，通知后端建图 |
| `tracking()` | 相机追踪：优化 `cam_rot_delta`/`cam_trans_delta`/曝光参数，最小化光度+深度误差 |
| `is_keyframe()` | 关键帧判定：基于位移量 + 可见区域重叠率（Szymkiewicz-Simpson 系数） |
| `add_to_window()` | 滑动窗口管理：维护局部关键帧窗口，移除冗余/低重叠帧 |
| `add_new_keyframe()` | 关键帧添加：对单目模式进行深度估计与不确定性建模 |
| `sync_backend()` | 同步后端结果：接收更新后的高斯模型 + 可见性 + 优化后的关键帧位姿 |

### 3. 后端建图 — `utils/slam_backend.py`

**类 `BackEnd(mp.Process)`**：运行在独立子进程中，负责高斯地图的创建与优化。

核心方法：

| 方法 | 功能 |
|------|------|
| `run()` | 主循环：等待 `backend_queue` 消息，处理 init / keyframe / color_refinement / stop |
| `initialize_map()` | 地图初始化：对首帧进行多轮渲染-优化迭代，执行高斯增密与修剪 |
| `map()` | 滑动窗口建图：对窗口内关键帧 + 随机采样历史帧进行联合优化；周期性增密/修剪/不透明度重置 |
| `color_refinement()` | 颜色精炼：26000 轮迭代，仅优化颜色（L1+SSIM），不改变几何 |
| `add_next_kf()` | 从关键帧点云扩展高斯：`GaussianModel.extend_from_pcd_seq()` |

**消息协议**（通过 `backend_queue` 接收）：

| 消息 | 触发 |
|------|------|
| `["init", frame_idx, viewpoint, depth_map]` | 系统初始化 |
| `["keyframe", frame_idx, viewpoint, window, depth_map]` | 新关键帧 |
| `["pause"]` / `["unpause"]` | GUI 暂停/恢复 |
| `["color_refinement"]` | 评估模式颜色精炼 |
| `["stop"]` | 停止进程 |

### 4. 2D 高斯场景表示 — `gaussian_splatting/scene/gaussian_model.py`

**类 `GaussianModel`**：管理 2D Gaussian 的所有属性与操作。

参数张量：

| 属性 | 维度 | 说明 |
|------|------|------|
| `_xyz` | (N, 3) | 2D 位置 |
| `_features_dc` | (N, 1, 3) | 零阶球谐系数 (RGB) |
| `_features_rest` | (N, 15, 3) | 高阶球谐系数 (视角相关颜色) |
| `_scaling` | (N, 2) | 2DGS 缩放 (2 维) 或 2DGS 缩放 (3 维) |
| `_rotation` | (N, 4) | 四元数旋转 |
| `_opacity` | (N, 1) | 不透明度 |
| `unique_kfIDs` | (N,) | 每个高斯所属的关键帧 ID |
| `n_obs` | (N,) | 每个高斯被观测到的次数 |

关键操作：

| 方法 | 功能 |
|------|------|
| `create_pcd_from_image_and_depth()` | 从 RGB-D 图像生成点云 → 初始化高斯 |
| `extend_from_pcd_seq()` | 从新关键帧点云扩展高斯集合 |
| `densify_and_prune()` | 基于梯度增密 / 基于尺度/不透明度修剪 |
| `reset_opacity()` | 重置所有高斯不透明度 |
| `prune_points()` | 根据 mask 删除高斯 |

### 5. 可微渲染器 — `gaussian_splatting/gaussian_renderer/__init__.py`

基于 **diff-surfel-rasterization**（2DGS 光栅化器）的可微渲染。

**函数 `render()`**：

1. 配置 `GaussianRasterizationSettings`（内参、外参、图像尺寸等）
2. 将 2D 高斯投影到屏幕空间
3. 执行 alpha 混合光栅化
4. 返回 `render_pkg`：渲染图像 / 深度图 / 不透明度 / 可见性过滤 / 视空间点 / 触达计数

### 6. 相机模型 — `utils/camera_utils.py`

**类 `Camera(nn.Module)`**：可微相机模型，关键参数：

- `R, T`：世界到相机的外参（可优化）
- `cam_rot_delta, cam_trans_delta`：位姿的李代数增量参数（追踪/BA 优化变量）
- `exposure_a, exposure_b`：曝光补偿参数
- `Fx, Fy, Cx, Cy, FoVx, FoVy`：内参
- `grad_mask`：梯度遮罩（排除图像边缘或无纹理区域）

### 7. 数据集 — `utils/dataset.py`

数据加载器，支持多种数据格式：

| 解析器类 | 数据格式 | 传感器类型 |
|---------|---------|-----------|
| `ReplicaParser` | Replica 数据集 | RGB-D |
| `TUMParser` | TUM-RGBD 数据集 | RGB-D / 单目 |
| `EurocParser` | EuRoC MAV 数据集 | 双目 |
| `RealsenseParser` | Intel RealSense 实时流 | RGB-D |

### 8. 损失函数 — `utils/slam_utils.py`

| 函数 | 用途 | 计算内容 |
|------|------|---------|
| `get_loss_tracking()` | 前端追踪 | 光度误差 (L1+SSIM) + 深度误差 |
| `get_loss_mapping()` | 后端建图 | 光度误差 + 深度误差 + 初始化专用项 |

### 9. 位姿工具 — `utils/pose_utils.py`

李代数 `se(3)` 操作：

| 函数 | 功能 |
|------|------|
| `SE3_exp(tau)` | 李代数到变换矩阵的指数映射 |
| `SO3_exp(theta)` | SO(3) 指数映射 |
| `update_pose()` | 使用优化后的 delta 更新相机外参，返回收敛标志 |

### 10. 评估工具 — `utils/eval_utils.py`

| 函数 | 功能 |
|------|------|
| `eval_ate()` | 绝对轨迹误差 (ATE)，使用 evo 库 |
| `eval_rendering()` | 渲染质量评估：PSNR / SSIM / LPIPS |
| `save_gaussians()` | 导出高斯模型为 .ply 文件 |

### 11. 可视化 GUI — `gui/`

| 文件 | 功能 |
|------|------|
| `slam_gui.py` | 主 GUI 窗口，OpenGL 渲染循环，键盘交互 |
| `gui_utils.py` | GUI 数据结构 (`GaussianPacket`, `ParamsGUI`) |
| `gl_render/render_ogl.py` | OpenGL 椭球体渲染器 (2D 高斯可视化) |
| `gl_render/util_gau.py` | 高斯模型与 OpenGL buffer 的转换 |
| `gl_render/util.py` | 通用 OpenGL 工具函数 |

### 12. 其他工具 — `utils/`

| 文件 | 功能 |
|------|------|
| `config_utils.py` | 层次化 YAML 配置加载（base_config → scene_config 合并） |
| `multiprocessing_utils.py` | 多进程辅助（`FakeQueue` 无 GUI 模式 / `clone_obj` 跨进程拷贝） |
| `point_utils.py` | 深度图转法线图 (`depth_to_normal`) |
| `logging_utils.py` | 统一日志输出 |

### 13. 配置系统 — `configs/`

```
configs/
├── mono/tum/          # 单目模式 (TUM 数据集)
│   ├── base_config.yaml
│   ├── fr1_desk.yaml / fr2_xyz.yaml / fr3_office.yaml
├── rgbd/
│   ├── replica/       # RGB-D 模式 (Replica 数据集)
│   │   ├── base_config.yaml
│   │   ├── office0~4.yaml / room0~2.yaml  (多进程版)
│   │   └── office0~4_sp.yaml / room0~2_sp.yaml (单进程版)
│   └── tum/           # RGB-D 模式 (TUM 数据集)
│       ├── base_config.yaml
│       └── fr1_desk.yaml / fr2_xyz.yaml / fr3_office.yaml
├── stereo/euroc/      # 双目模式 (EuRoC 数据集)
│   ├── base_config.yaml
│   └── mh02.yaml
└── live/              # 实时模式 (RealSense)
    ├── realsense.yaml
    └── realsense_rgbd.yaml
```

### 14. 子模块 — `submodules/`

| 子模块 | 说明 |
|--------|------|
| `diff-surfel-rasterization` | 2D Gaussian Splatting 可微光栅化器，支持位姿梯度反传 |
| `simple-knn` | CUDA 加速的 KNN 距离计算，用于高斯初始化时的尺度估计 |

## 整体数据流

```
1. 加载阶段
   数据集 ──→ 遍历帧 ──→ Camera 对象 ──→ Frontend 主循环

2. 初始化阶段 （仅单目模式）
   首帧 ──→ FrontEnd.initialize() ──→ backend_queue["init"]
   ──→ BackEnd.reset() → add_next_kf() → initialize_map()
   ──→ 多轮渲染+优化+增密 ──→ frontend_queue["init"] → FrontEnd.sync_backend()

3. 追踪-建图循环 （每帧迭代）
   ┌─ Frontend ──────────────────────────────────────────────────┐
   │  Camera.init_from_dataset(frame)                             │
   │  → tracking(): 渲染+优化位姿 (100轮)                          │
   │  → is_keyframe(): 判定是否添加关键帧                           │
   │  → 若关键帧: add_new_keyframe() → request_keyframe()         │
   │      └─ backend_queue.put(["keyframe", ...]) ──────────────┐ │
   └────────────────────────────────────────────────────────────┘ │
                                                                  │
   ┌─ Backend ───────────────────────────────────────────────────▼┘
   │  backend_queue.get() → "keyframe"
   │  → add_next_kf() 扩展高斯
   │  → map(current_window): 滑动窗口联合优化
   │     渲染所有关键帧 + 随机历史帧
   │     → 反向传播 → 增密/修剪/不透明度重置
   │     → 优化高斯属性 + 关键帧位姿 (BA)
   │  → push_to_frontend()
   │      └─ frontend_queue.put(["keyframe", gaussians, visibility, keyframes])
   │
   └──────────────────────────────────────────────────────────────┐
                                                                  │
   ┌─ Frontend ──────────────────────────────────────────────────▼┘
   │  frontend_queue.get() → "keyframe" → sync_backend()
   │  → 更新 gaussians, occ_aware_visibility, 关键帧位姿
   │  → 继续下一帧...
   └──────────────────────────────────────────────────────────────

4. 可视化通道 （并行，每 10 轮追踪迭代）
   Frontend ──→ q_main2vis.put(GaussianPacket)
   ──→ GUI OpenGL 渲染高斯 + 相机轨迹 + 当前帧

5. 评估阶段 （--eval 标志）
   追踪完成后 ──→ eval_ate() (ATE 轨迹误差)
   ──→ backend_queue["color_refinement"] → 26000轮颜色精炼
   ──→ eval_rendering() (PSNR/SSIM/LPIPS) → wandb 记录

进程间队列汇总:
┌──────────────────┬──────────┬──────────────────┐
│ 队列              │ 方向      │ 数据              │
├──────────────────┼──────────┼──────────────────┤
│ backend_queue     │ FE → BE  │ 命令 + 关键帧数据  │
│ frontend_queue    │ BE → FE  │ 同步后的高斯/位姿   │
│ q_main2vis        │ 主 → GUI │ GaussianPacket   │
│ q_vis2main        │ GUI → 主 │ 暂停/恢复控制      │
└──────────────────┴──────────┴──────────────────┘
```

## 运行

```bash
# 单目
python slam.py --config configs/mono/tum/fr3_office.yaml

# RGB-D
python slam.py --config configs/rgbd/tum/fr3_office.yaml
python slam.py --config configs/rgbd/replica/office0.yaml

# 双目
python slam.py --config configs/stereo/euroc/mh02.yaml

# 评估模式
python slam.py --config configs/mono/tum/fr3_office.yaml --eval

# 实时 (RealSense)
python slam.py --config configs/live/realsense.yaml
```


