
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from typing import List, Dict, Tuple, Optional, Literal, Any
import os
import csv
import numpy as np
from pathlib import Path
import cv2
import json

def to_np_rgb3(img) -> np.ndarray:
    """
    把输入（可能是 torch.Tensor / numpy，RGB 或 RGBA，HWC 或 CHW，uint8 或 float[0,1]）统一成：
      - numpy uint8
      - H×W×3 (RGB)
    """
    # -> numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # 常见：HWC 或 CHW
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[2] not in (3, 4):
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))

    # 如果是 float，将 [0,1] 或 [0,255] 统一到 uint8
    if np.issubdtype(img.dtype, np.floating):
        m = float(img.max()) if img.size else 1.0
        if m <= 1.0 + 1e-6:
            img = (img * 255.0).clip(0, 255)
        img = img.astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # 如果是 RGBA，丢弃 alpha
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    assert img.ndim == 3 and img.shape[2] == 3, f"Expect HxWx3 after sanitize, got {img.shape}"
    return img

class DinoGlobalEncoder:
    """Minimal global encoder using DINOv2 via torch.hub."""
    def __init__(self, pre_load_dinov2, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pre_load_dinov2
        self.prep = T.Compose([
            T.ToPILImage(),
            T.Resize(518, antialias=True),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        rgb = to_np_rgb3(rgb)
        x = self.prep(rgb).unsqueeze(0).to(self.device)
        feat = self.model(x)  # [1, D]
        feat = F.normalize(feat, dim=-1).cpu().numpy()[0].astype(np.float32)
        return feat

    @torch.inference_mode()
    def extract_patch_tokens(self, rgb: np.ndarray):
        """
        返回：
        - tokens_grid: (Hp, Wp, D)  仅 patch tokens（不含 CLS）
        - cls_token:   (D,)         最后一层 CLS token
        - (Hp, Wp):    patch 网格尺寸

        预处理分辨率固定 518×518，ViT/14 -> 37×37 网格。
        """
        rgb = to_np_rgb3(rgb)
        x = self.prep(rgb).unsqueeze(0).to(self.device)  # [1,3,518,518]

        # 取最后一层输出，并请求返回 CLS
        outs = self.model.get_intermediate_layers(x, n=1, return_class_token=True)
        last = outs[-1]

        if isinstance(last, tuple):
            # 兼容一种实现：返回 (patch_tokens, cls_token)
            # shapes: patch_tokens [B, N, D], cls_token [B, D]
            patch_tokens, cls_tok = last
        else:
            # 兼容另一种实现：返回 [B, N+1, D]，第 0 个是 CLS
            patch_tokens = last[:, 1:, :]         # [B, N, D]
            cls_tok = last[:, 0, :]               # [B, D]

        # 由 patch size 推 Hp, Wp
        ps = self.model.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        Hp = Wp = int(round(518 / ps))            # 518/14 ≈ 37

        # reshape 成网格
        patch_tokens = patch_tokens[0]            # [N, D]
        tokens_grid = patch_tokens[: Hp*Wp].reshape(Hp, Wp, -1)  # [Hp,Wp,D]

        # 取 CLS，压成 (D,)
        cls_token = cls_tok[0]                    # [D]

        return (
            cls_token.detach().cpu().numpy().astype(np.float32),
            tokens_grid.detach().cpu().numpy().astype(np.float32),
            (Hp, Wp),
        )



def build_explore_log_npz_from_jsonl(
    jsonl_path: str,
    save_dir: str,
    encoder,                      # 要求：feat = encoder.encode(rgb_uint8)
    fname: str = "explore_log.npz",
    sort_by_frame_id: bool = True,
):
    """
    生成 explore_log.npz:
      - frame_ids: [N] int32
      - poses_xyz: [N,3] float32        (与 jsonl pose.x/y/z 一致)
      - yaws:      [N] float32          (与 jsonl pose.yaw 一致)
      - feats:     [N,D] float16        (DINOv2特征)
      - quats:     [N,3] float32        (相机rotation xyz；从 jsonl pose.quat=[w,x,y,z] 取 x,y,z)
    """
    jsonl_path = Path(jsonl_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读 jsonl ----
    items = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    if sort_by_frame_id:
        items.sort(key=lambda x: int(x["frame_id"]))

    frame_ids = []
    poses_xyz = []
    yaws = []
    quats_xyz = []
    feats = []

    for idx, it in enumerate(items):
        fid = int(it["frame_id"])
        rgb_path = it["rgb_path"]
        pose = it["pose"]

        x = float(pose["x"]); y = float(pose["y"]); z = float(pose["z"])
        yaw = float(pose["yaw"])

        # jsonl quat 是 Habitat 顺序: [w, x, y, z]
        qw, qx, qy, qz = pose["quat"]
        qx = float(qx); qy = float(qy); qz = float(qz)

        frame_ids.append(fid)
        poses_xyz.append([x, y, z])
        yaws.append(yaw)
        quats_xyz.append([qx, qy, qz])   # 只存 rotation xyz（不含 w）

        # ---- 读图 & encode ----
        img_path = Path(rgb_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing rgb_path: {img_path}")

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)

        feat = encoder.encode(rgb)   # 你要求直接调用
        feat = np.asarray(feat)

        # 兼容 [D] 或 [1,D]
        if feat.ndim == 2 and feat.shape[0] == 1:
            feat = feat[0]
        if feat.ndim != 1:
            raise RuntimeError(f"encoder.encode output must be [D] or [1,D], got {feat.shape}")

        feats.append(feat.astype(np.float32))

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx+1}/{len(items)} frames...")

    # ---- 打包保存 ----
    frame_ids = np.asarray(frame_ids, dtype=np.int32)
    poses_xyz = np.asarray(poses_xyz, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    quats_xyz = np.asarray(quats_xyz, dtype=np.float32)
    feats = np.stack(feats, axis=0).astype(np.float16)  # [N,D]

    out_path = save_dir / fname
    np.savez_compressed(
        out_path,
        frame_ids=frame_ids,
        poses_xyz=poses_xyz,
        yaws=yaws,
        feats=feats,
        quats=quats_xyz,
    )

    print("Saved:", out_path)
    print("Shapes:",
          "frame_ids", frame_ids.shape,
          "poses_xyz", poses_xyz.shape,
          "yaws", yaws.shape,
          "feats", feats.shape,
          "quats", quats_xyz.shape)
# ==========================
# 使用示例（你把 encoder 初始化好即可）
# ==========================
if __name__ == "__main__":
    root_dir = "/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52"
    jsonl_path = f"{root_dir}/rgbd_export/frames_meta.jsonl"
    rgb_dir  = f"{root_dir}/rgbd_export/rgb_unique"
    save_dir = f"{root_dir}/rgbd_export"


    preload_dinov2 = torch.hub.load('/home/wangbo/codes/BSC-Nav/third-party/dinov2', "dinov2_vitl14_reg", source='local').to('cuda')

    encoder = DinoGlobalEncoder(pre_load_dinov2=preload_dinov2, device="cuda")

    if encoder is None:
        raise RuntimeError("Please initialize your DINOv2 encoder and assign to `encoder`.")

    build_explore_log_npz_from_jsonl(
        jsonl_path=jsonl_path,
        save_dir=save_dir,
        encoder=encoder,
        fname="explore_log.npz",
        sort_by_frame_id=True,
    )