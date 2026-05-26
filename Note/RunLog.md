# MG-Nav 操作流水

## 2026-05-22 摸底与 Note 初始化

### 仓库状态

- 路径：`/home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav`
- 上游 remote：`git@github.com:bo-wang-up/MG-Nav.git`（HEAD `73330ec`）
- **尚无** `mgnav` / `navdp` conda 环境
- `third-party/` 仅有 `habitat-lab`、`NavDP`；**缺** `dinov2`、`Grounded-SAM-2`
- **无** `data_episode/`、`memory/` 预构建图
- **无** `hm3d_annotated_basis.scene_dataset_config.json`（需单独下载 semantic-configs）

### 本机可复用数据（与 UniGoal 相同根）

| 资源 | 路径 |
|------|------|
| HM3D val 场景 | `~/data/processed/unigoal_datasets/scene_datasets/hm3d_v0.2/val/` |
| Instance-ImageNav v3 | `~/data/processed/unigoal_datasets/instance_imagenav/hm3d/v3/val/val.json.gz` |
| Demo 场景 00810 | `.../val/00810-CrMo8WxCyVb/CrMo8WxCyVb.basis.glb` ✓ |

### 已做（Note 侧）

1. 新增 `Note/README.md`、`ReplicationNotes.md`、`ChangeLog.md`、`CodeSummary.md`
2. `Note/results/launch_construct_graph.py` — 路径替换 + `torch.hub` 补丁
3. `Note/results/launch_mg_nav.py` — HM3D 路径默认注入
4. `wangbo_localization.py` — `localization` 的 shim（供 quick 脚本）
5. 待执行：数据软链、克隆第三方、conda 环境、NavDP 权重

### Git

- `benjamin` → `git@github.com:BenjaminChen1024/MG-Nav.git`，`main` @ `babd181`（2026-05-26 推送）

### 待办

- [x] `data_episode` → unigoal instance_imagenav
- [x] `data/hm3d` → hm3d_v0.2
- [x] `git clone` dinov2、Grounded-SAM-2
- [x] `conda create -n mgnav` + habitat-sim 0.3.3（需 `-c conda-forge -c aihabitat`）
- [x] habitat-lab / habitat-baselines editable 安装
- [ ] `pip install -r requirements.txt`（后台，日志 `Note/results/pip_install_mgnav.log`）
- [ ] 下载 `hm3d-val-semantic-configs-v0.2.tar`（~30KB 级配置包）
- [ ] `git clone` dinov2、Grounded-SAM-2 + 权重
- [ ] `conda create -n mgnav` + habitat-sim **0.3.3**
- [ ] `conda create -n navdp` + NavDP checkpoint
- [ ] 单场景探索 → 建图 → NavDP 服务 → 5 episode 冒烟
