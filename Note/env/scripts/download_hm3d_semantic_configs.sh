#!/usr/bin/env bash
# 下载 HM3D val 语义配置包（需 Matterport 账号，不跑仿真）
set -euo pipefail

DATA_ROOT="${MGNAV_DATA_ROOT:-$HOME/data/processed/mg-nav_datasets}"
DEST="$DATA_ROOT/scene_datasets/hm3d_v0.2"
TAR="hm3d-val-semantic-configs-v0.2.tar"
URL="https://api.matterport.com/resources/habitat/$TAR"
TMP="${TMPDIR:-/tmp}/$TAR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

if [[ -f "$DEST/hm3d_annotated_basis.scene_dataset_config.json" ]]; then
  echo "[OK] 已存在 $DEST/hm3d_annotated_basis.scene_dataset_config.json"
  exit 0
fi

if [[ -z "${MATTERPORT_USERNAME:-}" || -z "${MATTERPORT_PASSWORD:-}" ]]; then
  echo "请设置 Matterport 凭据后重试，例如："
  echo "  export MATTERPORT_USERNAME=你的邮箱"
  echo "  export MATTERPORT_PASSWORD=你的密码"
  echo "  bash Note/env/scripts/download_hm3d_semantic_configs.sh"
  echo ""
  echo "或浏览器登录后手动下载："
  echo "  $URL"
  echo "  tar -xf $TAR -C $DEST"
  exit 1
fi

mkdir -p "$DEST"
echo "[download] $URL"
curl -fsSL -u "$MATTERPORT_USERNAME:$MATTERPORT_PASSWORD" -o "$TMP" "$URL" || {
  echo "[ERROR] 下载失败。若返回 Unauthorized，通常表示："
  echo "  1) 邮箱/密码错误，或"
  echo "  2) 尚未通过 HM3D 研究数据集申请（与 Matterport 普通注册不同）"
  echo "  申请: https://matterport.com/habitat-matterport-3d-research-dataset"
  rm -f "$TMP"
  exit 1
}
if head -c 1 "$TMP" | grep -q '{' 2>/dev/null || file "$TMP" | grep -q JSON; then
  echo "[ERROR] 响应为 JSON 而非 tar（多为 Unauthorized）:"
  head -c 200 "$TMP"; echo
  rm -f "$TMP"
  exit 1
fi
tar -xf "$TMP" -C "$DEST"
rm -f "$TMP"
echo "[OK] $(ls -la "$DEST/hm3d_annotated_basis.scene_dataset_config.json")"
