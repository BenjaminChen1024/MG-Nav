#!/usr/bin/env bash
# 下载 MG-Nav 所需模型权重，并在结束时做完整性校验
# 强制重下: FORCE=1 bash Note/env/scripts/download_weights.sh
set -eo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG="$REPO/Note/env/logs/run_$(date +%Y%m%d)/download_weights.log"
mkdir -p "$(dirname "$LOG")"

FORCE="${FORCE:-0}"
[[ "${1:-}" == "--force" ]] && FORCE=1

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mgnav 2>/dev/null || true

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T')] download_weights start (FORCE=$FORCE)"

SAM2_CKPT="$REPO/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
GDINO_DIR="$REPO/third-party/Grounded-SAM-2/grounding-dino-tiny"
NAVDP_CKPT="$REPO/third-party/NavDP/checkpoints/checkpoint-43956navdp-onlyproj.ckpt"
SAM2_LARGE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
NAVDP_GDRIVE_ID="1m3dr3PKgKRADErC61y2aTneMOYWozljU"

mkdir -p "$(dirname "$SAM2_CKPT")" "$(dirname "$NAVDP_CKPT")" "$GDINO_DIR"

_pick_proxy() {
  local _p
  for _p in "${MGNAV_PROXY_PORT:-}" 17897 7897; do
    [[ -z "$_p" ]] && continue
    if curl -sS -m 5 -x "http://127.0.0.1:${_p}" -o /dev/null https://www.google.com 2>/dev/null; then
      echo "$_p"
      return 0
    fi
  done
  return 1
}

_need_download() {
  local path="$1" min_bytes="$2"
  [[ "$FORCE" == "1" ]] && return 0
  [[ ! -f "$path" ]] && return 0
  local sz
  sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
  [[ "$sz" -lt "$min_bytes" ]] && return 0
  return 1
}

# --- SAM2 ---
if _need_download "$SAM2_CKPT" 800000000; then
  echo "[download] SAM2.1 large ..."
  rm -f "$SAM2_CKPT"
  curl -fL --retry 3 --retry-delay 5 -o "$SAM2_CKPT" "$SAM2_LARGE_URL"
else
  echo "[skip] SAM2 exists $(du -h "$SAM2_CKPT" | cut -f1)"
fi

# --- GDINO ---
if [[ "$FORCE" == "1" ]] || [[ ! -f "$GDINO_DIR/config.json" ]] || \
   [[ ! -f "$GDINO_DIR/pytorch_model.bin" && ! -f "$GDINO_DIR/model.safetensors" ]]; then
  echo "[download] grounding-dino-tiny (HF) ..."
  [[ "$FORCE" == "1" ]] && rm -rf "$GDINO_DIR" && mkdir -p "$GDINO_DIR"
  pip install -q "huggingface_hub>=0.20"
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="IDEA-Research/grounding-dino-tiny",
    local_dir="${GDINO_DIR}",
)
print("gdino ok")
PY
else
  echo "[skip] grounding-dino-tiny exists"
fi

# --- NavDP ---
if _need_download "$NAVDP_CKPT" 3500000000; then
  echo "[download] NavDP ckpt (gdown) ..."
  pip install -q gdown
  rm -f "$NAVDP_CKPT"
  _proxy_port="$(_pick_proxy || true)"
  if [[ -n "$_proxy_port" ]]; then
    echo "[proxy] http://127.0.0.1:${_proxy_port}"
    export http_proxy="http://127.0.0.1:${_proxy_port}"
    export https_proxy="$http_proxy"
    export HTTP_PROXY="$http_proxy"
    export HTTPS_PROXY="$http_proxy"
    unset all_proxy ALL_PROXY
  else
    echo "[WARN] 无可用代理 (17897/7897)，gdown 可能失败"
  fi
  if ! gdown --fuzzy "https://drive.google.com/uc?id=${NAVDP_GDRIVE_ID}" -O "$NAVDP_CKPT"; then
    gdown "${NAVDP_GDRIVE_ID}" -O "$NAVDP_CKPT"
  fi
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
else
  echo "[skip] NavDP ckpt exists $(du -h "$NAVDP_CKPT" | cut -f1)"
fi

# --- integrity ---
echo "[verify] loading weights ..."
export REPO
python - <<'PY'
import os, sys
import torch

REPO = os.environ["REPO"]
SAM2 = f"{REPO}/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
GDINO = f"{REPO}/third-party/Grounded-SAM-2/grounding-dino-tiny"
NAVDP = f"{REPO}/third-party/NavDP/checkpoints/checkpoint-43956navdp-onlyproj.ckpt"
fail = []

def check_size(path, mb_min, name):
    if not os.path.isfile(path):
        fail.append(f"{name}: missing {path}")
        return
    sz = os.path.getsize(path) / (1024**2)
    if sz < mb_min:
        fail.append(f"{name}: too small ({sz:.0f} MiB < {mb_min} MiB)")

check_size(SAM2, 800, "SAM2")
check_size(NAVDP, 3500, "NavDP")

try:
    torch.load(SAM2, map_location="cpu", weights_only=False)
    print("[OK] SAM2 torch.load")
except Exception as e:
    fail.append(f"SAM2 load: {e}")

if not os.path.isfile(os.path.join(GDINO, "config.json")):
    fail.append("GDINO: no config.json")
else:
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        AutoProcessor.from_pretrained(GDINO)
        AutoModelForZeroShotObjectDetection.from_pretrained(GDINO)
        print("[OK] GDINO transformers")
    except Exception as e:
        fail.append(f"GDINO load: {e}")

try:
    ckpt = torch.load(NAVDP, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or len(ckpt) < 10:
        fail.append("NavDP: unexpected ckpt structure")
    else:
        print(f"[OK] NavDP torch.load ({len(ckpt)} keys)")
except Exception as e:
    fail.append(f"NavDP load: {e}")

if fail:
    print("[FAIL] integrity:", *fail, sep="\n  ")
    sys.exit(1)
print("[done] all weights verified")
PY

echo "[$(date '+%F %T')] download_weights finished OK"
