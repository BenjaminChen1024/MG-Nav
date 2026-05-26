# MG-Nav 本地变更记录

相对上游 [bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)。

## 2026-05-22

| 类型 | 路径 | 说明 |
|------|------|------|
| 新增 | `Note/**` | 复现文档与 launch 脚本 |
| 新增 | `wangbo_localization.py` | 指向 `localization`，兼容 quick/real_robot 导入 |
| 修改 | `.gitignore` | 增加 `data/`（本机软链不提交） |
| 未改 | 上游 `*.py` | 路径与 5090 适配均放在 `Note/results/launch_*.py` |

### Git 远程（与 UniGoal 一致）

| remote | URL | 用途 |
|--------|-----|------|
| `origin` | `git@github.com:bo-wang-up/MG-Nav.git` | 上游 |
| `benjamin` | `git@github.com:BenjaminChen1024/MG-Nav.git` | 本机 fork（`main` 已推送 `babd181`） |

### 计划中的软链（不提交大文件）

- `data_episode/imagenav/instance_imagenav_hm3d_v3` → `~/data/processed/unigoal_datasets/instance_imagenav/hm3d/v3`
- `data/hm3d` → `~/data/processed/unigoal_datasets/scene_datasets/hm3d_v0.2`
