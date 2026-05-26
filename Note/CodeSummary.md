# MG-Nav 代码入口摘要

## 主流程脚本

| 文件 | 作用 |
|------|------|
| `construct_graph_total.py` | 场景探索、语义分析、place graph 构建 |
| `run_navdp_follow_path_continuous_total.py` | 图 + NavDP 导航（含可视化） |
| `run_navdp_follow_path_continuous_total_quick.py` | 同上，关闭 metric/视频以加速 |
| `env.py` | Habitat-sim / Habitat-lab 封装、`get_objnav_env` |
| `localization.py` | `ImageNavGraphRobot`、DINO 定位 |
| `place_graph_builder_obs.py` | 图节点、语义特征、FPS 采样 |

## 第三方

| 目录 | 作用 |
|------|------|
| `third-party/habitat-lab` | Instance-ImageNav 配置与数据集 API |
| `third-party/NavDP` | 局部规划 RPC 服务与 client utils |
| `third-party/dinov2` | **需 clone** |
| `third-party/Grounded-SAM-2` | **需 clone** + `grounded_sam2_wrapper.py` |

## 运行时产物

```text
memory/<scene_id>/
  explore_log.npz
  floor_data.json
  place_graph_min{min_dis}_radius{radius}_floor{idx}.json
  obs/frames_meta.jsonl
  obs/sem/...
```

## 本机启动（推荐）

- 建图：`python Note/results/launch_construct_graph.py ...`
- 导航：`python Note/results/launch_mg_nav.py --quick ...`
