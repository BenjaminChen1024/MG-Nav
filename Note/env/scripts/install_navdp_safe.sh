#!/usr/bin/env bash
# 安装 conda 环境 navdp（NavDP RPC，暂不启动服务）
set -eo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$REPO/Note/env/logs/run_$(date +%Y%m%d)/install_navdp_safe.log"
mkdir -p "$(dirname "$LOG")"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
source "$(conda info --base)/etc/profile.d/conda.sh"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if ! conda env list | grep -qE '^navdp '; then
  log "create conda env navdp python=3.10 ..."
  conda create -n navdp python=3.10 -y >>"$LOG" 2>&1
fi
conda activate navdp

export MAX_JOBS=2
cd "$REPO/third-party/NavDP/baselines/navdp"
log "pip install navdp requirements ..."
nice -n 19 pip install -r requirements.txt >>"$LOG" 2>&1

log "upgrade torch cu128 for RTX 5090 (sm_120) ..."
pip uninstall -y torch torchvision torchaudio >>"$LOG" 2>&1 || true
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 --force-reinstall >>"$LOG" 2>&1
log "re-pin numpy/pillow (opencv needs numpy<2) ..."
pip install "numpy==1.26.0" "pillow==11.2.1" -q >>"$LOG" 2>&1
pip install einops -q >>"$LOG" 2>&1

log "=== verify navdp ==="
python -c "
import torch, numpy as np
print('navdp torch', torch.__version__, 'numpy', np.__version__)
x = torch.ones(2, device='cuda')
print('cuda ok', float(x.sum()))
" 2>&1 | tee -a "$LOG"
log "=== done navdp (start server only when ready) ==="
