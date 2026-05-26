# MG-Nav 复现笔记（本机）

上游：[bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)（Dual-Scale Visual Navigation via Sparse Spatial Memory）。

| 文档 | 说明 |
|------|------|
| [ReplicationNotes.md](./ReplicationNotes.md) | 复现结论、环境路线、排错 |
| [RunLog.md](./RunLog.md) | 操作流水（按时间追加） |
| [ChangeLog.md](./ChangeLog.md) | 相对上游的本地变更 |
| [CodeSummary.md](./CodeSummary.md) | 代码入口与数据流 |
| [results/launch_mg_nav.py](./results/launch_mg_nav.py) | 导航评测启动（路径/代理/导入补丁） |
| [results/launch_construct_graph.py](./results/launch_construct_graph.py) | 探索建图启动（不改上游 `.py`） |
| [env/setup_mgnav.md](./env/setup_mgnav.md) | `mgnav` 环境安装步骤 |
| [env/setup_navdp.md](./env/setup_navdp.md) | `navdp` 服务环境 |

## 快速命令（环境就绪后）

```bash
# 1) 探索 + 语义（单场景 demo，需 NavDP 未启动）
conda activate mgnav
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav
python Note/results/launch_construct_graph.py --explore_map --semantic_analyze

# 2) 建图
python Note/results/launch_construct_graph.py --construct_graph --visualize_graph \
  --floor_idx 0 --min_dis 1.0 --radius 0.5

# 3) NavDP 服务（另一终端，conda navdp）
# 见 env/setup_navdp.md

# 4) 导航评测（quick，无可视化）
python Note/results/launch_mg_nav.py --quick --rpc_port 6666 \
  --eval_episodes 5 --max_total_steps 500
```

数据根目录默认复用 UniGoal：`$HOME/data/processed/unigoal_datasets`（软链至 `data_episode/`、`data/hm3d`）。
