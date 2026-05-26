#!/usr/bin/env python
"""启动 MG-Nav 导航脚本（不改上游 .py）。

  --quick   使用 run_navdp_follow_path_continuous_total_quick.py（无视频、更快）
  默认      使用 run_navdp_follow_path_continuous_total.py

数据/模型路径通过环境变量与默认 argv 注入；需先启动 NavDP RPC 服务。
"""
from __future__ import annotations

import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get(
    "MGNAV_DATA_ROOT",
    os.path.expanduser("~/data/processed/unigoal_datasets"),
)
HM3D_ROOT = os.path.join(DATA_ROOT, "scene_datasets", "hm3d_v0.2")
EPISODE_JSON = os.path.join(
    DATA_ROOT,
    "instance_imagenav/hm3d/v3/val/val.json.gz",
)

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third-party", "Grounded-SAM-2"))
sys.path.insert(0, os.path.join(REPO, "third-party", "NavDP"))

for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

_use_quick = "--quick" in sys.argv
if _use_quick:
    sys.argv.remove("--quick")
    _target = "run_navdp_follow_path_continuous_total_quick.py"
else:
    _target = "run_navdp_follow_path_continuous_total.py"

# quick 版依赖 wangbo_localization；根目录已提供 shim
if _use_quick and not os.path.isfile(os.path.join(REPO, "wangbo_localization.py")):
    sys.stderr.write("缺少 wangbo_localization.py（应为 localization 的 shim）\n")
    sys.exit(1)

_defaults = []
if "--HM3D_SCENE_PREFIX" not in sys.argv:
    _defaults.extend(["--HM3D_SCENE_PREFIX", HM3D_ROOT])
if "--HM3D_EPISODE_PREFIX" not in sys.argv:
    rel_ep = os.path.join("data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz")
    if os.path.isfile(os.path.join(REPO, rel_ep)):
        _defaults.extend(["--HM3D_EPISODE_PREFIX", rel_ep])
    elif os.path.isfile(EPISODE_JSON):
        _defaults.extend(["--HM3D_EPISODE_PREFIX", EPISODE_JSON])

# 插入到脚本名之前
sys.argv[1:1] = _defaults

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

runpy.run_path(os.path.join(REPO, _target), run_name="__main__")
