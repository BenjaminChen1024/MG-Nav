#!/usr/bin/env bash
# 初始化 mg-nav_datasets：仅含解压后的真实数据；原始 tar/zip 在 ~/data/datasets/HM3D
set -euo pipefail

MG_ROOT="${MGNAV_DATA_ROOT:-/home/ial-chenzm/data/processed/mg-nav_datasets}"
UNIGOAL="${UNIGOAL_DATA_ROOT:-/home/ial-chenzm/data/processed/unigoal_datasets}"
HM3D_RAW="${HM3D_RAW:-/home/ial-chenzm/data/datasets/HM3D}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"

echo "[init] MG_ROOT=$MG_ROOT"
echo "[init] 原始包目录 HM3D_RAW=$HM3D_RAW（不在 mg-nav_datasets 内）"

mkdir -p "$MG_ROOT/scene_datasets" "$MG_ROOT/instance_imagenav/hm3d"
rm -rf "$MG_ROOT/raw"

# --- 场景 + 语义 config（真实目录）---
HM3D_SCENE="$MG_ROOT/scene_datasets/hm3d_v0.2"
UNI_HM3D="$UNIGOAL/scene_datasets/hm3d_v0.2"

if [[ -L "$HM3D_SCENE" ]]; then rm -f "$HM3D_SCENE"; fi
mkdir -p "$HM3D_SCENE"

if [[ -d "$UNI_HM3D/val" ]] && [[ -f "$UNI_HM3D/hm3d_annotated_basis.scene_dataset_config.json" ]]; then
  echo "[rsync] hm3d_v0.2 from unigoal (~6.6G) ..."
  rsync -a --info=stats2 "$UNI_HM3D/" "$HM3D_SCENE/"
elif [[ -f "$HM3D_RAW/hm3d-val-habitat-v0.2.tar" ]]; then
  mkdir -p "$HM3D_SCENE/val"
  [[ -f "$HM3D_SCENE/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb" ]] || {
    echo "[extract] $HM3D_RAW/hm3d-val-habitat-v0.2.tar -> val/"
    tar -xf "$HM3D_RAW/hm3d-val-habitat-v0.2.tar" -C "$HM3D_SCENE/val"
  }
  [[ -f "$HM3D_SCENE/hm3d_annotated_basis.scene_dataset_config.json" ]] || {
    echo "[extract] $HM3D_RAW/hm3d-val-semantic-configs-v0.2.tar"
    tar -xf "$HM3D_RAW/hm3d-val-semantic-configs-v0.2.tar" -C "$HM3D_SCENE"
  }
else
  echo "[ERROR] 无 hm3d 场景来源（unigoal 或 $HM3D_RAW/*.tar）"
  exit 1
fi

# --- Instance-ImageNav v3（真实目录）---
EP_V3="$MG_ROOT/instance_imagenav/hm3d/v3"
UNI_EP="$UNIGOAL/instance_imagenav/hm3d/v3"

if [[ -L "$EP_V3" ]]; then rm -f "$EP_V3"; fi
mkdir -p "$(dirname "$EP_V3")"

if [[ -f "$UNI_EP/val/val.json.gz" ]]; then
  echo "[rsync] instance_imagenav v3 ..."
  rsync -a "$UNI_EP/" "$EP_V3/"
elif [[ -f "$HM3D_RAW/instance_imagenav_hm3d_v3.zip" ]]; then
  echo "[extract] $HM3D_RAW/instance_imagenav_hm3d_v3.zip"
  TMP=$(mktemp -d)
  unzip -q "$HM3D_RAW/instance_imagenav_hm3d_v3.zip" -d "$TMP"
  SRC=$(find "$TMP" -name 'val.json.gz' -print -quit)
  SRC=$(dirname "$SRC")
  rsync -a "$SRC/" "$EP_V3/"
  rm -rf "$TMP"
else
  echo "[ERROR] 无 episode 来源"
  exit 1
fi

export MGNAV_DATA_ROOT="$MG_ROOT"
bash "$REPO/Note/env/scripts/setup_data.sh"

echo ""
echo "[done] $MG_ROOT（仅解压数据；raw 在 $HM3D_RAW）"
