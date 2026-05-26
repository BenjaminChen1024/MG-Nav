# MG-Nav 操作流水

> **主文档**：环境与命令见 `Note/env/README.md`；代码说明见 `CodeSummary.md`；复现问题见 `ReplicationNotes.md`。  
> 安装日志目录：`Note/env/logs/`（nohup / pip 历史，可随时删除）。

---

## 2026-05-22 — 摸底与 Note 初始化

### 仓库与远程

| 项 | 值 |
|----|-----|
| 路径 | `/home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav` |
| 上游 `origin` | `git@github.com:bo-wang-up/MG-Nav.git`（当时 HEAD `73330ec`） |
| 本机 `benjamin` | `git@github.com:BenjaminChen1024/MG-Nav.git`，`main` 曾推 `babd181` |

### 当时缺失项

- conda `mgnav` / `navdp` 未建
- `third-party/dinov2`、`Grounded-SAM-2` 未 clone
- `data_episode/`、`memory/` 无
- `hm3d_annotated_basis.scene_dataset_config.json` 未就位

### Note 初版产出

1. `ReplicationNotes.md`、`ChangeLog.md`、`CodeSummary.md`
2. `Note/results/launch_construct_graph.py` — 路径 + `torch.hub` 补丁
3. `Note/results/launch_mg_nav.py` — HM3D 默认路径
4. `Note/results/repro_imports.py` — quick 入口 `wangbo_localization` → `localization`
5. 原则：**不改上游根目录 `.py`**

---

## 2026-05-26 — 数据、环境、权重

### 数据布局定稿

| 用途 | 路径 |
|------|------|
| 真实解压数据 | `~/data/processed/mg-nav_datasets` |
| 原始 tar/zip | `~/data/datasets/HM3D/` |
| 仓库软链 | `data/hm3d`、`data_episode/...` → 上者 |

**已齐**：HM3D val 101 场景、ImageNav v3 `val.json.gz`、`hm3d_annotated_basis.scene_dataset_config.json`（自 `hm3d-val-semantic-configs-v0.2.tar` 解压）。

### 脚本（后迁至 `Note/env/scripts/`）

| 脚本 | 结果 |
|------|------|
| `init_mgnav_datasets.sh` | rsync / 初始化 mg-nav_datasets |
| `setup_data.sh` | 软链 OK |
| `install_mgnav_safe.sh` | habitat-sim 0.3.3 + habitat-lab editable |
| `install_navdp_safe.sh` | navdp 基础依赖 |
| `download_weights.sh` | SAM2 ✓；GDINO 经 `HF_ENDPOINT=hf-mirror.com` ✓ |
| NavDP ckpt | 直连 Google 超时；经代理 **17897** gdown **4.0G** ✓ |

### mgnav 环境细节

| 步骤 | 说明 |
|------|------|
| habitat-sim | `0.3.3` py3.9 headless bullet（conda-forge + aihabitat） |
| requirements | 大量 pip；CLIP 首次失败（`pkg_resources`）→ `setuptools<81` 后成功 |
| torch | 由 2.6+cu124 升为 **2.8.0+cu128**；重 pin `numpy==1.26.4` `pillow==10.4.0` |
| 5090 | CUDA matmul、GDINO 模型 `.cuda()` 通过 |

### navdp 环境（5090 复查后修补）

| 问题 | 处理 |
|------|------|
| `torch==2.2.2+cu121` | `no kernel image` on sm_120 → 升为 **2.11.0+cu128** |
| torch 升级带入 numpy 2.x | opencv 崩溃 → `numpy==1.26.0` |
| 缺 `einops` | `pip install einops` |
| NavDP_Agent 加载 | GPU0 占满 OOM；**GPU3** 上加载 checkpoint **成功** |

### dinov2 + Python 3.9

- `dinov2` main 使用 `float | None` 注解 → py3.9 `TypeError`
- 处理：`launch_construct_graph.py` 自动注入 `from __future__ import annotations`；`third-party/dinov2/dinov2/layers/{attention,block}.py` 已写入
- GPU3：`dinov2_vitl14_reg` torch.hub 加载 **OK**

### 其他兼容性

| 项 | 结论 |
|----|------|
| `import clip` 在 `habitat_sim` **之后** | **段错误**；须先 clip |
| 代理 7897 | 服务器未监听；**17897** 可用 |
| `verify_setup.sh` | 数据 + 权重 + 双环境 **ALL READY** |
| `memory/` | 仍空，待 construct_graph |

### 日志文件位置（已整理）

| 文件 | 说明 |
|------|------|
| `Note/env/logs/run_20260526/install_mgnav_safe.log` | mgnav 安装 |
| `Note/env/logs/run_20260526/install_navdp_safe.log` | navdp 安装 |
| `Note/env/logs/run_20260526/download_weights.log` | 权重下载 |
| `Note/env/logs/*.nohup.out` | 后台任务 stdout |
| `Note/env/logs/pip_install_mgnav*.log` | 早期 pip 尝试，可删 |

---

## 2026-05-26 — Note 目录整理

### 结构调整

| 变更 | 说明 |
|------|------|
| 删除 `Note/README.md` | 入口改为 `RunLog.md` + `env/README.md` |
| 合并 env 文档 | 删除 `setup_mgnav.md`、`setup_navdp.md`、`SETUP_CHECKLIST.md`、`DATA_SOURCES.md` → 单文件 `Note/env/README.md` |
| `Note/scripts/` → `Note/env/scripts/` | 安装/数据/权重脚本 |
| `requirements_no_clip.txt` → `Note/env/` | 环境配置 |
| 日志 → `Note/env/logs/` | nohup、pip 历史 |
| 保留 `Note/results/` | 仅 launch 与 `repro_imports.py` |

### 待执行（程序阶段）

- [ ] 单场景 `00810`：`launch_construct_graph.py --explore_map --semantic_analyze`
- [ ] 同场景 `--construct_graph`
- [ ] `navdp` 服务 + `launch_mg_nav.py --quick` 冒烟
- [ ] 全量 val 前确认 GPU 占用与 `memory/` 磁盘

---

## 常用命令（当前路径）

```bash
export MGNAV_DATA_ROOT=~/data/processed/mg-nav_datasets
bash Note/env/scripts/verify_setup.sh

conda activate mgnav
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
CUDA_VISIBLE_DEVICES=3 python Note/results/launch_construct_graph.py \
  --explore_map --semantic_analyze
```
