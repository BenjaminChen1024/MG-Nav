#!/usr/bin/env bash
# 从本机复现 UniGoal 时整理的 HuggingFace 镜像拉取 HM3D（无需 Matterport 登录）
# Dataset: https://huggingface.co/datasets/BenjaminChen1024/VLN_Dataset
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
HF_BASE="${HF_VLN_BASE:-https://huggingface.co/datasets/BenjaminChen1024/VLN_Dataset/resolve/main}"
HM3D_RAW="${HM3D_RAW:-$HOME/data/datasets/HM3D}"
DATA_ROOT="${MGNAV_DATA_ROOT:-$HOME/data/processed/mg-nav_datasets}"
SCENE_ROOT="$DATA_ROOT/scene_datasets/hm3d_v0.2"
EP_ROOT="$DATA_ROOT/instance_imagenav/hm3d/v3"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
mkdir -p "$SCENE_ROOT/val" "$EP_ROOT/val" "$(dirname "$SCENE_ROOT")"

dl() {
  local rel="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "[skip] $dest"
    return 0
  fi
  echo "[wget] $rel"
  wget -c "${HF_BASE}/${rel}" -O "$dest"
}

mkdir -p "$HM3D_RAW"

# --- 原始包下载到 ~/data/datasets/HM3D（不在 mg-nav_datasets）---
TAR_SCENE="$HM3D_RAW/hm3d-val-habitat-v0.2.tar"
dl "HM3D/hm3d-val-habitat-v0.2.tar" "$TAR_SCENE"

ZIP_EP="$HM3D_RAW/instance_imagenav_hm3d_v3.zip"
dl "HM3D/instance_imagenav_hm3d_v3.zip" "$ZIP_EP"

SEM_TAR="$HM3D_RAW/hm3d-val-semantic-configs-v0.2.tar"
[[ -f "$SEM_TAR" ]] || echo "[note] semantic-configs 需 Matterport 下载到 $HM3D_RAW"

# --- 解压到 mg-nav_datasets（或运行 init 脚本）---
if [[ ! -f "$SCENE_ROOT/val/00810-CrMo8WxCyVb/CrMo8WxCyVb.basis.glb" ]] && [[ -f "$TAR_SCENE" ]]; then
  echo "[extract] habitat -> $SCENE_ROOT/val/"
  mkdir -p "$SCENE_ROOT/val"
  tar -xf "$TAR_SCENE" -C "$SCENE_ROOT/val"
fi
if [[ -f "$SEM_TAR" ]] && [[ ! -f "$SCENE_ROOT/hm3d_annotated_basis.scene_dataset_config.json" ]]; then
  tar -xf "$SEM_TAR" -C "$SCENE_ROOT"
fi
if [[ ! -f "$EP_ROOT/val/val.json.gz" ]] && [[ -f "$ZIP_EP" ]]; then
  TMP=$(mktemp -d)
  unzip -q "$ZIP_EP" -d "$TMP"
  mkdir -p "$EP_ROOT"
  SRC=$(find "$TMP" -name 'val.json.gz' -print -quit)
  rsync -a "$(dirname "$SRC")/" "$EP_ROOT/"
  rm -rf "$TMP"
fi

# --- 仓库内软链 ---
bash "$REPO/Note/env/scripts/setup_data.sh"

echo ""
echo "[done] raw -> $HM3D_RAW ; 解压数据 -> $DATA_ROOT"
echo "  或: bash Note/env/scripts/init_mgnav_datasets.sh"
