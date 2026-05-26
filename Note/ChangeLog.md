# MG-Nav 本地变更记录

相对上游 [bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)。

---

## 2026-05-26 — Note 结构整理

| 类型 | 路径 | 说明 |
|------|------|------|
| 重组 | `Note/env/` | 合并环境文档为 `README.md`；脚本迁入 `env/scripts/`；日志迁入 `env/logs/` |
| 删除 | `Note/README.md` | 入口改为 `RunLog.md` |
| 删除 | `Note/env/setup_*.md`、`SETUP_CHECKLIST.md`、`DATA_SOURCES.md` | 内容并入 `env/README.md` |
| 移动 | `Note/scripts/*` → `Note/env/scripts/` | 更新脚本内 `REPO=../../..`、日志路径 |
| 移动 | `Note/requirements_no_clip.txt` → `Note/env/` | |
| 扩充 | `CodeSummary.md`、`RunLog.md`、`ReplicationNotes.md` | 逐文件说明与 5090 记录 |

---

## 2026-05-26 — 环境与 5090

| 类型 | 路径 | 说明 |
|------|------|------|
| 修改 | `Note/env/scripts/install_navdp_safe.sh` | 5090：torch cu128、numpy 1.26、einops |
| 修改 | `Note/env/scripts/install_mgnav_safe.sh` | setuptools<81、torch cu128、numpy/pillow pin |
| 修改 | `Note/env/scripts/download_weights.sh` | HF 镜像；NavDP 代理探测 7897/17897 |
| 修改 | `Note/env/scripts/verify_setup.sh` | navdp 5090 检查；`memory/` 信息项 |
| 修改 | `Note/results/launch_construct_graph.py` | dinov2 py3.9 `future annotations` 补丁 |
| 修改 | `third-party/dinov2/dinov2/layers/attention.py` | 同上（运行时/持久补丁） |
| 修改 | `third-party/dinov2/dinov2/layers/block.py` | 同上 |
| 数据 | `mg-nav_datasets` | HM3D val + ImageNav + semantic config 齐备 |
| 权重 | `third-party/NavDP/checkpoints/*.ckpt` | 约 4.0G，经代理 17897 下载 |

---

## 2026-05-22 — Note 初建

| 类型 | 路径 | 说明 |
|------|------|------|
| 新增 | `Note/**` | 复现文档、launch、安装脚本骨架 |
| 新增 | `Note/results/repro_imports.py` | quick 脚本 import 指到 `localization` |
| 新增 | `Note/results/launch_construct_graph.py` | 建图启动器 |
| 新增 | `Note/results/launch_mg_nav.py` | 导航启动器 |
| 修改 | `.gitignore` | 增加 `data/`（软链不提交） |
| 未改 | 上游根目录 `*.py` | 路径与 5090 适配在 launch / Note 脚本层 |

### Git 远程

| remote | URL | 用途 |
|--------|-----|------|
| `origin` | `git@github.com:bo-wang-up/MG-Nav.git` | 上游 |
| `benjamin` | `git@github.com:BenjaminChen1024/MG-Nav.git` | 本机 fork |

### 数据约定

- 真实数据：`~/data/processed/mg-nav_datasets/`
- 仓库软链：`Note/env/scripts/setup_data.sh`
