#!/usr/bin/env python
"""启动 construct_graph_total.py（不修改上游源码）。

在 import 前将作者机路径替换为本机 REPO / 数据路径，并补丁 torch.hub.load。
默认场景：00810-CrMo8WxCyVb（与上游 SCENE_ID_MAP 一致）。

用法示例：
  python Note/results/launch_construct_graph.py --explore_map --semantic_analyze
  python Note/results/launch_construct_graph.py --construct_graph --visualize_graph \\
      --floor_idx 0 --min_dis 1.0 --radius 0.5
"""
from __future__ import annotations

import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get(
    "MGNAV_DATA_ROOT",
    os.path.expanduser("~/data/processed/mg-nav_datasets"),
)
HM3D_ROOT = os.path.join(DATA_ROOT, "scene_datasets", "hm3d_v0.2")
DATASET_VAL = os.path.join(HM3D_ROOT, "val")
SCENE_CONFIG = os.path.join(HM3D_ROOT, "hm3d_annotated_basis.scene_dataset_config.json")

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third-party", "Grounded-SAM-2"))

for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_k, None)

if not os.path.isfile(SCENE_CONFIG):
    sys.stderr.write(
        f"[WARN] 缺少语义场景配置: {SCENE_CONFIG}\n"
        "  请下载 hm3d-val-semantic-configs-v0.2.tar 解压到 hm3d_v0.2/ 根目录。\n"
        "  见 Note/env/README.md\n"
    )

# 默认 CLI 覆盖（可被用户追加参数覆盖）
# 在 argv 中注入默认路径（仅当用户未显式指定）
def _inject_default(flag: str, value: str) -> None:
    if flag in sys.argv:
        return
    sys.argv[1:1] = [flag, value]

_inject_default("--dataset_dir", DATASET_VAL)
_inject_default("--scene_dataset_config_file", SCENE_CONFIG)

import torch  # noqa: E402


def _patch_dinov2_py39() -> None:
    """dinov2 main 使用 PEP604 类型注解，Python 3.9 需 future annotations。"""
    if sys.version_info >= (3, 10):
        return
    dino_root = os.path.join(REPO, "third-party", "dinov2", "dinov2")
    for rel in ("layers/attention.py", "layers/block.py"):
        p = os.path.join(dino_root, rel)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            txt = f.read()
        if "from __future__ import annotations" in txt:
            continue
        with open(p, "w", encoding="utf-8") as f:
            f.write("from __future__ import annotations\n" + txt)


_patch_dinov2_py39()

_orig_hub = torch.hub.load


def _patched_hub(repo_or_dir, model, *args, source="github", **kwargs):
    if isinstance(repo_or_dir, str) and (
        repo_or_dir.endswith("dinov2") or "/MG-Nav/" in repo_or_dir or "/BSC-Nav/" in repo_or_dir
    ):
        repo_or_dir = os.path.join(REPO, "third-party", "dinov2")
    return _orig_hub(repo_or_dir, model, *args, source=source, **kwargs)


torch.hub.load = _patched_hub

_src_path = os.path.join(REPO, "construct_graph_total.py")
with open(_src_path, "r", encoding="utf-8") as f:
    _src = f.read()

_replacements = {
    "/home/wangbo/codes/MG-Nav": REPO,
    "/nas_dataset/wangbo/HM3D/val/": DATASET_VAL + ("" if DATASET_VAL.endswith("/") else "/"),
    "/nas_dataset/wangbo/HM3D/hm3d_annotated_basis.scene_dataset_config.json": SCENE_CONFIG,
    "sam2_checkpoint = \"/home/wangbo/codes/MG-Nav/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt\"":
        'sam2_checkpoint = os.path.join(REPO, "third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")',
    "gdino_id = \"/home/wangbo/codes/MG-Nav/third-party/Grounded-SAM-2/grounding-dino-tiny\"":
        'gdino_id = os.path.join(REPO, "third-party/Grounded-SAM-2/grounding-dino-tiny")',
}
for old, new in _replacements.items():
    _src = _src.replace(old, new)

# GroundedSAM2 块需要 os
if "import os, math" in _src and "REPO = " not in _src:
    _src = _src.replace(
        "import os, math",
        f"import os, math\nREPO = {repr(REPO)}",
        1,
    )

_spec = importlib.util.spec_from_loader("construct_graph_total", loader=None)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["construct_graph_total"] = _mod
exec(compile(_src, _src_path, "exec"), _mod.__dict__)
