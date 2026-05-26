#!/usr/bin/env bash
# MG-Nav 仓库 data/ → mg-nav_datasets（真实数据在 DATA_ROOT，此处仅建软链）
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_ROOT="${MGNAV_DATA_ROOT:-$HOME/data/processed/mg-nav_datasets}"

echo "[data] REPO=$REPO"
echo "[data] DATA_ROOT=$DATA_ROOT"

if [[ ! -d "$DATA_ROOT/scene_datasets/hm3d_v0.2/val" ]]; then
  echo "[ERROR] 缺少 HM3D val 场景: $DATA_ROOT/scene_datasets/hm3d_v0.2/val"
  exit 1
fi

mkdir -p "$REPO/data_episode/imagenav" "$REPO/data"
ln -sfn "$DATA_ROOT/instance_imagenav/hm3d/v3" \
  "$REPO/data_episode/imagenav/instance_imagenav_hm3d_v3"
ln -sfn "$DATA_ROOT/scene_datasets/hm3d_v0.2" "$REPO/data/hm3d"

echo "[data] episode -> $(readlink -f "$REPO/data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz" 2>/dev/null || echo MISSING)"
echo "[data] scenes  -> $(readlink -f "$REPO/data/hm3d/val" 2>/dev/null || echo MISSING)"

SEM_CFG="$DATA_ROOT/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
if [[ -f "$SEM_CFG" ]]; then
  echo "[OK] semantic config: $SEM_CFG"
else
  echo "[PENDING] 需 Matterport 下载 hm3d-val-semantic-configs-v0.2.tar 并解压到:"
  echo "          $DATA_ROOT/scene_datasets/hm3d_v0.2/"
  echo "          见 Note/env/scripts/download_hm3d_semantic_configs.sh"
fi
