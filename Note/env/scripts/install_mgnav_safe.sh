#!/usr/bin/env bash
# 安装/补全 conda 环境 mgnav（5090：末尾升级 torch 2.7+cu128）
set -eo pipefail

LOCK=/tmp/mgnav_install.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "另一 mgnav 安装任务正在运行，退出。"
  exit 0
fi

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$REPO/Note/env/logs/run_$(date +%Y%m%d)/install_mgnav_safe.log"
mkdir -p "$(dirname "$LOG")"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
source "$(conda info --base)/etc/profile.d/conda.sh"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if ! conda env list | grep -qE '^mgnav '; then
  log "create conda env mgnav python=3.9 ..."
  conda create -n mgnav python=3.9 cmake=3.14.0 -y >>"$LOG" 2>&1
fi
conda activate mgnav

export MAX_JOBS=2
export MAKEFLAGS="-j2"

if ! python -c "import habitat_sim" 2>/dev/null; then
  log "install habitat-sim 0.3.3 ..."
  conda install -c conda-forge -c aihabitat -y \
    "habitat-sim=0.3.3=py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24" \
    >>"$LOG" 2>&1
fi

if ! python -c "import habitat" 2>/dev/null; then
  log "install habitat-lab editable ..."
  cd "$REPO/third-party/habitat-lab"
  nice -n 19 pip install -e habitat-lab -e habitat-baselines >>"$LOG" 2>&1
  cd "$REPO"
fi

log "pip base tools ..."
nice -n 19 pip install -U setuptools wheel setuptools-scm >>"$LOG" 2>&1

REQ_NO_CLIP="$REPO/Note/env/requirements_no_clip.txt"
grep -v '^clip @' "$REPO/requirements.txt" >"$REQ_NO_CLIP"

log "install requirements (except clip) ..."
nice -n 19 pip install -r "$REQ_NO_CLIP" >>"$LOG" 2>&1 || true

log "pin setuptools for CLIP build (pkg_resources) ..."
pip install "setuptools<81" -q >>"$LOG" 2>&1

log "install CLIP (--no-build-isolation) ..."
nice -n 19 pip install --no-build-isolation \
  "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1" \
  >>"$LOG" 2>&1

log "pin pillow for habitat-sim ..."
pip install "pillow==10.4.0" "numpy==1.26.4" -q >>"$LOG" 2>&1

log "upgrade torch 2.7+cu128 for RTX 5090 (sm_120) ..."
nice -n 19 pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 >>"$LOG" 2>&1

log "re-pin numpy/pillow for habitat-sim after torch ..."
pip install "numpy==1.26.4" "pillow==10.4.0" -q >>"$LOG" 2>&1

log "=== verify mgnav ==="
python -c "
import habitat_sim, habitat, torch, numpy as np
print('habitat_sim', habitat_sim.__version__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.ones(2, device='cuda')
    print('cuda tensor ok', float(x.sum()))
print('numpy', np.__version__)
" 2>&1 | tee -a "$LOG"

log "=== done mgnav install ==="
