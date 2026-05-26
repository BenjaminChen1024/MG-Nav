# MG-Nav 复现心得

> 本机：**4× RTX 5090（sm_120）**；数据根 `~/data/processed/mg-nav_datasets`。  
> 上游：[bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)（arXiv:2511.22609）。  
> **操作流水**：[RunLog.md](./RunLog.md) · **环境**：[env/README.md](./env/README.md) · **代码**：[CodeSummary.md](./CodeSummary.md)

---

## 1. 总体结论（2026-05-26）

| 阶段 | 状态 |
|------|------|
| 数据 + 权重 | ✓ 齐备 |
| `mgnav` / `navdp` conda | ✓ `verify_setup.sh` 全绿 |
| 5090 CUDA | ✓ 两环境均已用 **cu128** torch 验证 |
| `memory/` 建图 | ✗ 未跑 |
| 端到端导航评测 | ✗ 未跑 |

MG-Nav 是 **探索建图 + NavDP RPC + 图规划** 三件套，比 UniGoal/SG-Nav 更重；上游大量**作者绝对路径**，本机用 `Note/results/launch_*.py` 规避改码。

---

## 2. 与 UniGoal / SG-Nav 差异

| 项目 | MG-Nav | UniGoal（本机） |
|------|--------|-----------------|
| habitat-sim | **0.3.3** | 0.2.3 / 0.2.4 |
| habitat-lab | 仓库 `third-party` v3 | 旧子模块 |
| 感知 | DINOv2 + SAM2.1 + GDINO-tiny | SAM + GroundingDINO 等 |
| 局部规划 | **NavDP**（独立 conda + ckpt） | 内置策略 |
| 评测前置 | 每场景 **`memory/<scene>/`** | 可直接 ImageNav |

---

## 3. RTX 5090（sm_120）— 必改项

上游 README / `requirements.txt` 按 **torch 2.6+cu124**（mgnav）、**2.2.2+cu121**（navdp）编写，在 5090 上会出现：

```text
NVIDIA GeForce RTX 5090 ... not compatible ...
RuntimeError: no kernel image is available for execution on the device
```

| 环境 | 上游锁定 | 本机实测可用 |
|------|----------|--------------|
| `mgnav` | torch 2.6.0+cu124 | **2.8.0+cu128** |
| `navdp` | torch 2.2.2+cu121 | **2.11.0+cu128** |

安装脚本 `Note/env/scripts/install_*_safe.sh` 已在末尾强制 cu128；**勿**在 navdp 里仅 `pip install -r requirements.txt` 而不升级 torch。

**附带约束**：

- habitat-sim 要求 `numpy==1.26.4`、`pillow==10.4.0` — torch 升级后必须重 pin
- navdp 升级 torch 后须 `numpy==1.26.0`，否则 opencv 与 numpy2 冲突
- navdp 需额外 **`einops`**（不在原 requirements.txt）

---

## 4. 仍存在的复现风险

### 4.1 代码与 Python 版本

| 问题 | 影响 | 缓解 |
|------|------|------|
| `dinov2` main 使用 py3.10 语法 | mgnav 为 py3.9，hub 加载失败 | `launch_construct_graph.py` + 已 patch 的 `dinov2/layers/*.py` |
| `import habitat_sim` 后再 `import clip` | 段错误 | 先 clip；construct 通常不用 clip |
| `construct_graph_total.py` 顶层加载大模型 | import 即占 GPU | 用空闲 GPU；避免与 UniGoal 抢卡 |
| `run_*_quick.py` 引用 `wangbo_localization` | ImportError | `launch_mg_nav.py` + `repro_imports.py` |

### 4.2 第三方安装不完整

| 组件 | 现象 | 建议 |
|------|------|------|
| `sam2` | 仅 `sys.path` 下可 import | 首次报错则 `cd third-party/Grounded-SAM-2 && pip install -e .` |
| `grounding_dino` CUDA 扩展 | 官方需 nvcc | 本机 GDINO 走 **transformers** 本地 tiny，未编译原版 CUDA 算子 |
| `dinov2` 浅 clone | 仅 1 commit | 一般够用；需换版本则 `git fetch --unshallow` |

### 4.3 运行依赖

| 项 | 说明 |
|----|------|
| NavDP 服务 | 必须先 `navdp_server_geometry.py --port` 与评测脚本一致 |
| `memory/<scene>/` | 无图 JSON 则导航无法开始 |
| GPU 内存 | NavDP ckpt + GSAM + DINOv2 同卡易 OOM；建图与 RPC 建议分卡 |
| 代理 | 服务器 `7897` 常未开；Google 用 **17897**（见 RunLog） |

### 4.4 与上游 README 不一致处（本机刻意为之）

- **不修改**根目录 `construct_graph_total.py` 等 — 路径靠 launch 字符串替换
- **habitat-sim 0.3.3** vs 部分文档写 0.2.x — 以本仓库 `third-party/habitat-lab` 为准
- **torch 版本**高于上游 pin — 为 5090 必要代价

---

## 5. 数据与权重（当前）

均已就位，详见 `verify_setup.sh`：

- HM3D val、`hm3d_annotated_basis.scene_dataset_config.json`
- Instance-ImageNav v3
- `sam2.1_hiera_large.pt`、`grounding-dino-tiny/`、NavDP `checkpoint-43956navdp-onlyproj.ckpt`

**不需**为 val 评测单独下载 `hm3d-val-semantic-annots`（*.semantic.glb）；MG-Nav 语义来自 RGB + GSAM。

---

## 6. 推荐执行顺序

```mermaid
flowchart LR
  A[verify_setup] --> B[launch_construct_graph explore]
  B --> C[semantic_analyze]
  C --> D[construct_graph]
  D --> E[navdp server]
  E --> F[launch_mg_nav quick]
```

Demo 场景：`00810-CrMo8WxCyVb`（上游 `SCENE_ID_MAP` 默认）。

```bash
# 环境
bash Note/env/scripts/verify_setup.sh

# 建图（mgnav，空闲 GPU）
CUDA_VISIBLE_DEVICES=3 python Note/results/launch_construct_graph.py \
  --explore_map --semantic_analyze
CUDA_VISIBLE_DEVICES=3 python Note/results/launch_construct_graph.py \
  --construct_graph --floor_idx 0 --min_dis 1.0 --radius 0.5

# NavDP（navdp，另一终端）
CUDA_VISIBLE_DEVICES=3 python third-party/NavDP/baselines/navdp/navdp_server_geometry.py \
  --port 6666 --checkpoint third-party/NavDP/checkpoints/checkpoint-43956navdp-onlyproj.ckpt

# 评测
CUDA_VISIBLE_DEVICES=3 python Note/results/launch_mg_nav.py --quick \
  --rpc_port 6666 --eval_episodes 5 --max_total_steps 500
```

---

## 7. 文档索引

| 文档 | 用途 |
|------|------|
| [RunLog.md](./RunLog.md) | 按日期的操作、排错、命令 |
| [env/README.md](./env/README.md) | 安装、数据、第三方、代理 |
| [CodeSummary.md](./CodeSummary.md) | 仓库逐文件说明 |
| [ChangeLog.md](./ChangeLog.md) | 本地 git / Note 变更 |
