#!/usr/bin/env bash
# 检查环境/数据/权重是否就绪（不跑 construct_graph / nav）
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${MGNAV_DATA_ROOT:-$HOME/data/processed/mg-nav_datasets}"
FAIL=0

ok()  { echo "  [OK]   $*"; }
pend(){ echo "  [TODO] $*"; FAIL=1; }

echo "=== MG-Nav setup checklist ==="

# --- data ---
echo "[data]"
if [[ -f "$REPO/data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz" ]] || \
   [[ -f "$DATA_ROOT/instance_imagenav/hm3d/v3/val/val.json.gz" ]]; then
  ok "Instance-ImageNav v3 val.json.gz"
else
  pend "episode 数据"
fi
if [[ -d "$DATA_ROOT/scene_datasets/hm3d_v0.2/val/00810-CrMo8WxCyVb" ]]; then
  ok "HM3D demo scene 00810"
else
  pend "HM3D val 场景目录"
fi
if [[ -f "$DATA_ROOT/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json" ]]; then
  ok "hm3d_annotated_basis.scene_dataset_config.json"
else
  pend "HM3D semantic configs (Matterport)"
fi

# --- third-party code ---
echo "[code]"
[[ -d "$REPO/third-party/dinov2" ]] && ok "dinov2" || pend "third-party/dinov2"
[[ -d "$REPO/third-party/Grounded-SAM-2" ]] && ok "Grounded-SAM-2" || pend "Grounded-SAM-2"

# --- weights ---
echo "[weights]"
[[ -f "$REPO/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt" ]] \
  && ok "sam2.1_hiera_large.pt" || pend "SAM2.1 large"
[[ -f "$REPO/third-party/Grounded-SAM-2/grounding-dino-tiny/config.json" ]] \
  && ok "grounding-dino-tiny" || pend "grounding-dino-tiny"
[[ -f "$REPO/third-party/NavDP/checkpoints/checkpoint-43956navdp-onlyproj.ckpt" ]] \
  && ok "NavDP checkpoint" || pend "NavDP .ckpt"

# --- conda mgnav ---
echo "[env mgnav]"
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE '^mgnav '; then
  conda activate mgnav
  python -c "import habitat_sim; import habitat; import torch" 2>/dev/null \
    && ok "mgnav: habitat_sim + habitat + torch" || pend "mgnav python imports"
  python -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit('no cuda')
cap=torch.cuda.get_device_capability()
x=torch.ones(2,device='cuda')
# sm_120 needs torch 2.7+cu128
v=torch.__version__
if cap>=(12,0) and not (v.startswith('2.7') or v.startswith('2.8')):
    raise SystemExit('5090 needs torch 2.7+/cu128, got '+v)
" 2>/dev/null && ok "mgnav: 5090 cuda tensor" || pend "mgnav: torch cuda / 5090 版本"
else
  pend "conda env mgnav"
fi

# --- conda navdp ---
echo "[env navdp]"
if conda env list | grep -qE '^navdp '; then
  conda activate navdp
  python -c "import torch; import numpy as np; assert np.__version__.startswith('1.')" 2>/dev/null \
    && ok "navdp: torch + numpy<2" || pend "navdp imports / numpy"
  python -c "
import torch
if not torch.cuda.is_available():
    raise SystemExit('no cuda')
cap=torch.cuda.get_device_capability()
v=torch.__version__
if cap>=(12,0) and not (v.startswith('2.7') or v.startswith('2.8') or v.startswith('2.11')):
    raise SystemExit('5090 needs torch 2.7+/cu128, got '+v)
torch.ones(2, device='cuda')
" 2>/dev/null && ok "navdp: 5090 cuda tensor" || pend "navdp: torch cuda / 5090"
else
  pend "conda env navdp"
fi

# --- runtime artifacts (not blockers for env setup) ---
echo "[runtime]"
if [[ -d "$REPO/memory" ]] && [[ -n "$(ls -A "$REPO/memory" 2>/dev/null)" ]]; then
  ok "memory/ 已有建图结果"
else
  echo "  [INFO] memory/ 为空：需先跑 construct_graph"
fi

echo "=== summary: $([[ $FAIL -eq 0 ]] && echo 'ALL READY' || echo 'PENDING ITEMS ABOVE') ==="
exit $FAIL
