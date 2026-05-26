# MG-Nav 复现心得（2026-05-22 起）

> 本机：**4× RTX 5090（sm_120）**；数据与 UniGoal 共用 `~/data/processed/unigoal_datasets`。  
> 上游：[bo-wang-up/MG-Nav](https://github.com/bo-wang-up/MG-Nav)（arXiv:2511.22609）。

---

## 1. 总体结论（当前阶段）

MG-Nav 是 **两阶段 + 双环境** 系统，比 UniGoal/SG-Nav **更重**：

1. **探索建图**（`construct_graph_total.py`）：DINOv2 + Grounded-SAM-2 + habitat-sim **0.3.3**，为每个场景生成 `memory/<scene_id>/`。
2. **NavDP 服务**（独立 conda `navdp`，Python 3.10）：RPC 提供 point-goal 策略。
3. **图导航评测**（`run_navdp_follow_path_continuous_total*.py`）：在 place graph 上规划，调用 NavDP 执行。

上游代码含大量**作者机绝对路径**（如 `/home/*/codes/...`、`/nas_dataset/*/`）；**不修改 `.py`** 时，需用 `Note/results/launch_*.py` 做字符串替换或 CLI 覆盖。

**本日状态：环境尚未安装完成，链路未冒烟。**

---

## 2. 与 UniGoal / SG-Nav 的差异

| 项目 | MG-Nav | UniGoal（本机已跑通） |
|------|--------|----------------------|
| habitat-sim | **0.3.3** | 0.2.3 / 0.2.4 |
| habitat-lab | 仓库内 `third-party` v3 系 | 旧版子模块 |
| 额外模型 | DINOv2、SAM2.1、Grounding-DINO-tiny | SAM、GroundingDINO（GSA） |
| 局部规划 | **NavDP**（第二环境 + checkpoint） | 内置 RL/启发式 |
| 评测前置 | 需先 **per-scene 探索建图** | 直接 Instance-ImageNav |

---

## 3. 数据与配置

### 3.1 已有（复用 UniGoal）

- HM3D val 场景：`scene_datasets/hm3d_v0.2/val/<scene_id>/*.basis.glb`
- Episode：`instance_imagenav/hm3d/v3/val/val.json.gz`（1000 ep）

### 3.2 仍缺

| 资源 | 用途 | 获取方式 |
|------|------|----------|
| `hm3d_annotated_basis.scene_dataset_config.json` | 语义 / 建图 | Matterport `hm3d-val-semantic-configs-v0.2.tar` 解压到 `hm3d_v0.2/` |
| `third-party/dinov2` | 全局特征 | `git clone` facebookresearch/dinov2 |
| `third-party/Grounded-SAM-2` | 检测分割 | IDEA-Research/Grounded-SAM-2 + 权重 |
| NavDP `.ckpt` | RPC 服务 | README Google Drive 链接 |
| `memory/<scene>/` | 导航 | 本地跑探索建图生成 |

---

## 4. 本机环境路线（计划）

| 环境 | Python | 关键依赖 |
|------|--------|----------|
| `mgnav` | 3.9 | habitat-sim **0.3.3** headless+bullet、torch **2.6**（requirements）、habitat-lab editable |
| `navdp` | 3.10 | torch 2.2.2、Flask RPC（`third-party/NavDP/baselines/navdp`） |

5090 注意：与 UniGoal 相同，优先 **torch 2.x + cu12.8**；若 0.3.3 预编译包仅 cu12.4，先按 README 安装再测 `torch.cuda`。

---

## 5. 推荐执行顺序

```mermaid
flowchart LR
  A[数据软链 + semantic config] --> B[conda mgnav + 第三方]
  B --> C[探索 explore_map]
  C --> D[建图 construct_graph]
  D --> E[conda navdp + checkpoint]
  E --> F[NavDP server :6666]
  F --> G[launch_mg_nav 评测]
```

单场景 demo：`00810-CrMo8WxCyVb`（上游默认 `SCENE_ID_MAP`）。

---

## 6. 已知坑（来自读码）

1. **`construct_graph_total.py` 在 import 时加载 DINOv2/GSAM** — 权重路径必须在 launch 中替换。
2. **`run_*_quick.py` 使用非公开 localization 模块名** — 仓库仅有 `localization.py`；由 `launch_mg_nav.py` + `repro_imports.py` 在运行时改 import。
3. **NavDP 必须先起服务**，否则 `socket` 连接失败。
4. **GPU 占用**：UniGoal 长跑时避免与 MG-Nav 抢同卡（默认 launch 用 GPU 3）。

---

## 7. 文档索引

- 操作流水：[RunLog.md](./RunLog.md)
- 变更：[ChangeLog.md](./ChangeLog.md)
- 代码入口：[CodeSummary.md](./CodeSummary.md)
