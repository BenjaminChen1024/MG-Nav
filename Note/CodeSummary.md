# MG-Nav 代码文件说明

> 上游：[bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)。本机通过 `Note/results/launch_*.py` 做路径与 import 补丁，**不修改**上游根目录 `.py` 逻辑（除 `third-party/dinov2` 的 py3.9 兼容行，见 RunLog）。

---

## 仓库根目录

### `construct_graph_total.py`

**角色**：HM3D 仿真下的**探索 → 语义分析 → place graph 构建**主入口。

**依赖**：`env.NavEnv`、`place_graph_builder_obs.PlaceGraphBuilder`、模块级预加载 **DINOv2**（`torch.hub`）与 **GroundedSAM2**（硬编码作者路径 `/home/wangbo/codes/MG-Nav/...`）。

**主要阶段**（由 CLI 开关控制）：

- `--explore_map`：在场景中运动并写 `memory/<scene_id>/explore_log.npz`
- `--semantic_analyze`：对观测帧跑 GSAM，写 `obs/sem/`、`frames_meta.jsonl`
- `--construct_graph` / `--visualize_graph`：聚类节点、连边，输出 `place_graph_min{dis}_radius{r}_floor{f}.json`

**本机**：用 `Note/results/launch_construct_graph.py` 替换路径、补丁 `torch.hub`、dinov2 py3.9。

---

### `construct_graph_real_robot.py`

**角色**：与 `construct_graph_total.py` 同结构，面向**真机 / BSC-Nav** 路径（`BSC-Nav`、`wangbo_place_graph_builder_obs` 等 import）。

**本机**：一般不跑；若跑需自行改路径或仿照 `launch_construct_graph.py` 做 launch。

---

### `env.py`

**角色**：Habitat **Instance-ImageNav** 环境封装。

**要点**：

- `NavEnv`：基于 `habitat_sim` + `habitat-lab`，管理 RGB/深度、动作、top-down map
- `get_objnav_env(...)`：按场景 ID、数据集路径构造评测用 env
- 注册自定义 measurement / action（lookup、collision 等）

**被谁调用**：`construct_graph_total.py`、`run_navdp_follow_path_continuous_*.py`。

---

### `place_graph_builder_obs.py`

**角色**：**Place graph 构建器**（约 3.5k 行），核心算法模块。

**职责**：

- 从 `explore_log.npz` / 在线探索读取位姿与 RGB
- DBSCAN / FPS 等生成节点；边权与几何关系
- 与 GSAM 语义特征、`SemanticSaver` 结合，写 `floor_data.json`、图 JSON
- 可选调用 `NavEnv` 做在线探索（`env` 可注入）

**依赖**：sklearn、habitat 可视化、`env.NavEnv`（可选）。

---

### `localization.py`

**角色**：**图像导航定位**（`ImageNavGraphRobot` 等）。

**要点**：

- `DinoGlobalEncoder`：用预加载 DINOv2 提全局特征
- 在 place graph 上匹配当前帧与目标图像、节点归属
- 与 `run_navdp_*` 配合做重定位与 success 判定

**被谁调用**：`run_navdp_follow_path_continuous_total.py` 及 quick 变体。

---

### `grounded_sam2_wrapper.py`

**角色**：**Grounded SAM 2** 薄封装（检测 + 分割），供建图语义与导航侧可选检测。

**API**：`GroundedSAM2(sam2_checkpoint, gdino_id)` → `.detect(image, text_prompt)` 等。

**副本**：同文件应位于 `third-party/Grounded-SAM-2/`（上游 README 要求 copy）。

---

### `run_navdp_follow_path_continuous_total.py`

**角色**：**图 + NavDP 导航评测**（带可视化、metric、视频）。

**流程**：

1. `get_objnav_env` 加载 episode
2. 读 `memory/<scene>/place_graph_*.json`
3. 图上的 A* / 节点路径 + **HTTP/RPC 调 NavDP**（`utils_tasks.client_utils`）
4. `ImageNavGraphRobot` 做图像目标与节点匹配

**依赖**：`third-party/NavDP` 的 adapter、**需先启动** `navdp_server_geometry.py`。

---

### `run_navdp_follow_path_continuous_total_quick.py`

**角色**：与 `total.py` 相同导航逻辑，**关闭**慢路径（`env.get_metric()`、视频、部分 path 更新）以加速 benchmark。

**本机**：`launch_mg_nav.py --quick` 会将 `wangbo_localization` import 指到 `localization`（见 `repro_imports.py`）。

---

### `run_navdp_follow_path_continuous_real_robot.py`

**角色**：真机版 NavDP×图导航（socket/实机接口，作者 NAS 路径）。

**本机**：需实机与路径改造，复现 HM3D 评测可忽略。

---

### `run_navdp_follow_path_continuous_real_robot_easy.py`

**角色**：真机导航简化版（更少依赖或更短流程，与上类似）。

---

### `rosbag_rgbd_load.py`

**角色**：从 **ROS2 bag** 读取 RGB-D / 里程计（`rclpy`），供真机管线使用。

**本机**：无 ROS 环境时可忽略。

---

### `requirements.txt`

**角色**：上游 **mgnav** pip 锁定（含 `habitat` 系、`torch==2.6.0`、`transformers` git 等）。

**本机**：经 `install_mgnav_safe.sh` 安装后**升级 torch cu128**；CLIP 单独装；实际 pin 见 `Note/env/README.md`。

---

### `README.md`

**角色**：上游官方安装与运行说明（作者路径、双 conda、Google Drive 权重）。

---

### `.gitignore`

**角色**：忽略 `data/`（软链）、`memory/`、权重等大文件。

---

## `real_robot/`（真机数据工具，HM3D 复现可跳过）

| 文件 | 作用 |
|------|------|
| `RGBD_data_load.py` | 从 rosbag2 导出 RGB-D 帧 |
| `RGB_data_load.py` | 仅 RGB 话题导出 |
| `RGB_cam_pos.py` | 从 bag 读相机位姿 |
| `RGB_unique.py` | 按位姿/时间对 RGB 去重 |
| `csv_to_jsonl.py` | CSV 轨迹转 jsonl（供建图/导航读） |
| `explore_log_npz.py` | 与探索日志相关的特征/导出工具 |
| `video_generation.py` | 将导出图像序列合成 mp4 |

均为**硬编码 NAS 路径**的离线脚本，不参与 HM3D 标准评测链路。

---

## `data/`、`data_episode/`（软链）

| 路径 | 指向 |
|------|------|
| `data/hm3d` | `mg-nav_datasets/scene_datasets/hm3d_v0.2` |
| `data_episode/imagenav/instance_imagenav_hm3d_v3` | `.../instance_imagenav/hm3d/v3` |

由 `Note/env/scripts/setup_data.sh` 创建。

---

## `memory/`（运行时生成）

每场景一目录，典型文件：

- `explore_log.npz` — 探索轨迹
- `floor_data.json` — 楼层信息
- `place_graph_min*_radius*_floor*.json` — 导航用图
- `obs/frames_meta.jsonl`、`obs/sem/*` — 语义缓存

**导航前必须先对目标场景跑完建图。**

---

## `Note/` 文档与启动器

| 文件 | 作用 |
|------|------|
| `RunLog.md` | 按时间的操作与排错记录（主索引） |
| `ChangeLog.md` | 相对上游的本地变更 |
| `CodeSummary.md` | 本文件 |
| `ReplicationNotes.md` | 复现结论与已知问题 |
| `env/README.md` | 环境、数据、第三方、安装 |
| `env/requirements_no_clip.txt` | 去掉 clip 行的 requirements，供安装脚本 |
| `env/scripts/*.sh` | 安装 / 数据 / 权重 / 校验 |
| `env/logs/` | 历史安装日志（可删） |
| `results/launch_construct_graph.py` | 建图：路径替换、`torch.hub`、dinov2 py3.9 |
| `results/launch_mg_nav.py` | 导航：HM3D 路径、GPU、`repro_imports` |
| `results/repro_imports.py` | quick 脚本 import 补丁 |

---

## `Note/env/scripts/` 各脚本

| 脚本 | 作用 |
|------|------|
| `install_mgnav_safe.sh` | 创建/补全 `mgnav`；habitat-sim 0.3.3；habitat-lab editable；5090 torch |
| `install_navdp_safe.sh` | 创建/补全 `navdp`；5090 torch；numpy/einops |
| `download_weights.sh` | SAM2、grounding-dino-tiny、NavDP ckpt |
| `setup_data.sh` | 仓库数据软链 |
| `init_mgnav_datasets.sh` | 初始化 `mg-nav_datasets` |
| `download_hm3d_from_hf.sh` | HF 镜像下载场景与 episode |
| `download_hm3d_semantic_configs.sh` | Matterport semantic-configs |
| `verify_setup.sh` | 一键检查数据/权重/环境/5090 |

---

## `third-party/`（摘要，安装见 `Note/env/README.md`）

| 目录 | 作用 |
|------|------|
| `habitat-lab/` | Habitat 3.x 配置、Instance-ImageNav 数据集 API |
| `NavDP/` | 策略网络 + `navdp_server_geometry.py` + habitat adapters |
| `dinov2/` | DINOv2 torch.hub 本地仓库 |
| `Grounded-SAM-2/` | SAM2 + Grounding DINO + `grounded_sam2_wrapper.py` |

---

## 推荐调用链（HM3D val）

```text
launch_construct_graph.py  →  memory/<scene>/
navdp_server_geometry.py   →  RPC :6666
launch_mg_nav.py --quick     →  SR/SPL 评测
```
