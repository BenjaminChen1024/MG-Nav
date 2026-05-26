# MG-Nav 环境与安装

> 数据根：`$HOME/data/processed/mg-nav_datasets`（`MGNAV_DATA_ROOT`）  
> 本机：**4× RTX 5090（sm_120）**；与上游 README 的 torch 2.6+cu124 **不直接兼容**，见下文。

## 目录结构

```text
Note/env/
  README.md                 # 本文件
  requirements_no_clip.txt  # pip 安装用（去掉 clip git 行，由脚本单独装）
  scripts/                  # 安装、数据、权重脚本
  logs/                     # 安装/下载日志（nohup、pip 历史，可删）
```

## 一键安装顺序（tmux 推荐）

```bash
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

bash Note/env/scripts/init_mgnav_datasets.sh   # 或已有 mg-nav_datasets 可跳过
bash Note/env/scripts/setup_data.sh            # 仓库内软链

bash Note/env/scripts/install_mgnav_safe.sh    # 较久
bash Note/env/scripts/install_navdp_safe.sh
bash Note/env/scripts/download_weights.sh      # Google Drive 需代理见下

bash Note/env/scripts/verify_setup.sh
```

HM3D 语义配置（`hm3d_annotated_basis.scene_dataset_config.json`）若缺失：

```bash
# 方式 A：本机已有 tar
tar -xf ~/data/datasets/HM3D/hm3d-val-semantic-configs-v0.2.tar -C \
  "$MGNAV_DATA_ROOT/scene_datasets/hm3d_v0.2"

# 方式 B：Matterport API（需账号）
export MATTERPORT_USERNAME=... MATTERPORT_PASSWORD=...
bash Note/env/scripts/download_hm3d_semantic_configs.sh
```

## Conda 环境

| 环境 | Python | 用途 | 5090 要点 |
|------|--------|------|-----------|
| `mgnav` | 3.9 | 建图、`construct_graph`、habitat-sim **0.3.3** | 末尾升级 **torch 2.7+/cu128**（本机 2.8.0+cu128）；pin `numpy==1.26.4` `pillow==10.4.0` |
| `navdp` | 3.10 | NavDP RPC `navdp_server_geometry.py` | **勿**用 requirements 里 `torch==2.2.2+cu121`；脚本会升到 **cu128** + `numpy==1.26.0` + `einops` |

### mgnav 补充

- CLIP：`setuptools<81` 后 `pip install --no-build-isolation clip@git+...`
- **dinov2**：main 分支含 py3.10 类型注解；`launch_construct_graph.py` 会为 `dinov2/layers/{attention,block}.py` 注入 `from __future__ import annotations`
- **勿** `import habitat_sim` 后再 `import clip`（段错误）；需 CLIP 时先 import clip
- HF 权重：`export HF_ENDPOINT=https://hf-mirror.com`（`download_weights.sh` 已默认）

### navdp 启动示例

```bash
conda activate navdp
cd third-party/NavDP
CUDA_VISIBLE_DEVICES=3 python baselines/navdp/navdp_server_geometry.py \
  --port 6666 \
  --checkpoint ./checkpoints/checkpoint-43956navdp-onlyproj.ckpt
```

## 数据布局

| 路径 | 内容 |
|------|------|
| `~/data/processed/mg-nav_datasets/` | 真实数据（HM3D val、ImageNav v3、semantic config） |
| `~/data/datasets/HM3D/` | 原始 tar/zip（可选） |
| 仓库 `data/hm3d` | 软链 → `.../scene_datasets/hm3d_v0.2` |
| 仓库 `data_episode/imagenav/instance_imagenav_hm3d_v3` | 软链 → `.../instance_imagenav/hm3d/v3` |

可选 HF 镜像：[BenjaminChen1024/VLN_Dataset](https://huggingface.co/datasets/BenjaminChen1024/VLN_Dataset) → `download_hm3d_from_hf.sh`

## 第三方仓库（需 clone + 权重）

| 目录 | 来源 | 权重 / 说明 |
|------|------|-------------|
| `third-party/habitat-lab` | 随仓库 | `pip install -e habitat-lab -e habitat-baselines` |
| `third-party/NavDP` | 随仓库 | `checkpoints/checkpoint-43956navdp-onlyproj.ckpt`（Google Drive） |
| `third-party/dinov2` | `facebookresearch/dinov2` | torch.hub 本地加载 `dinov2_vitl14_reg` |
| `third-party/Grounded-SAM-2` | `IDEA-Research/Grounded-SAM-2` | `checkpoints/sam2.1_hiera_large.pt`；`grounding-dino-tiny/`（HF） |
| | | 将仓库根目录 `grounded_sam2_wrapper.py` 复制到 `Grounded-SAM-2/`（已存在则可跳过） |

首次跑建图若 `import sam2` 失败，在 `mgnav` 中：`cd third-party/Grounded-SAM-2 && pip install -e .`（按官方 README，可能需编译 grounding_dino）。

## 代理（Google Drive）

- `.bashrc` 默认 `127.0.0.1:7897` 在**服务器**上常未监听
- 本机 SSH 转发可用 **`17897`**：`export MGNAV_PROXY_PORT=17897` 后跑 `download_weights.sh`

## 脚本索引

| 脚本 | 作用 |
|------|------|
| `init_mgnav_datasets.sh` | 从 unigoal_datasets rsync 到 mg-nav_datasets |
| `setup_data.sh` | 仓库 `data/`、`data_episode/` 软链 |
| `install_mgnav_safe.sh` | conda mgnav + habitat-sim 0.3.3 + 依赖 + 5090 torch |
| `install_navdp_safe.sh` | conda navdp + 5090 torch |
| `download_weights.sh` | SAM2 / GDINO / NavDP ckpt |
| `download_hm3d_from_hf.sh` | HF 镜像拉 HM3D + ImageNav |
| `download_hm3d_semantic_configs.sh` | Matterport semantic-configs |
| `verify_setup.sh` | 数据 / 权重 / 环境 / 5090 CUDA 检查 |

## 验证

```bash
export MGNAV_DATA_ROOT=~/data/processed/mg-nav_datasets
bash Note/env/scripts/verify_setup.sh
```

日志：`Note/env/logs/`（不重要可整目录删除）。
