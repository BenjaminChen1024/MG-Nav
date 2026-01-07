
import os
import json
import math
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Literal, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.cluster import DBSCAN


from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height, to_grid, get_topdown_map, colorize_topdown_map
from habitat.utils.visualizations import maps
import cv2 

from itertools import combinations
import pycocotools.mask as mask_util

from collections import deque

import magnum as mn
from habitat_sim.utils.common import quat_to_magnum

# Try to import user's NavEnv; if not present, online exploration won't run.
try:
    from env import NavEnv
except Exception as e:
    NavEnv = None
    print("[WARN] Could not import NavEnv from env.py. You can pass an env instance to builder.build().", e)

def get_sim_cam_mat_with_fov(h, w, fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat

def _save_png_rgba(path_png: str, rgba_img: np.ndarray):
    os.makedirs(os.path.dirname(path_png) or ".", exist_ok=True)
    cv2.imwrite(path_png, rgba_img)

def _save_pdf_from_png(path_pdf: str, rgba_png_path: str):
    # 用 ReportLab 将 PNG 原尺寸嵌入 PDF（保留透明）
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.utils import ImageReader

    # 以像素做页面尺寸（1 pt ~ 1 px 假定；如需指定 DPI，可等比缩放）
    import PIL.Image as PILImage
    with PILImage.open(rgba_png_path) as im:
        Wpx, Hpx = im.size

    os.makedirs(os.path.dirname(path_pdf) or ".", exist_ok=True)
    c = rl_canvas.Canvas(path_pdf, pagesize=(Wpx, Hpx))
    img = ImageReader(rgba_png_path)
    # (0,0) 到 (Wpx, Hpx) 填满一页；无缩放无裁剪
    c.drawImage(img, 0, 0, width=Wpx, height=Hpx, mask='auto')
    c.showPage()
    c.save()

# -----------------------------
# Data classes for graph
# -----------------------------
@dataclass
class PlaceNode:
    id: int

    # —— 空间优先图用到的字段 ——
    center: Dict[str, float]                    # {"x","y","z"}
    radius: float
    members: List[int] = field(default_factory=list)        # 该节点覆盖的 frame_ids
    keyframes: List[int] = field(default_factory=list)      # 该节点的 keyframe frame_ids
    object_list: List[Dict[str, Any]] = field(default_factory=list)

EDGE_TYPE2CODE = {
    "temporal": 0,
    "geometric": 1,
    # "semantic": 2,   # 以后需要可再加
}
def encode_edge_type(t: str) -> int:
    return EDGE_TYPE2CODE.get(str(t).lower(), -1)

@dataclass
class Edge:
    source_id: int
    target_id: int
    type: str                          # "temporal" | "geometric" | ...
    delta_pose: Dict[str, float]       # {"dx":..., "dz":..., "dyaw":...}
    distance: float
    count: int = 1                     # 这条转移在轨迹中出现的次数（聚合后的权重）
    type_code: int = field(init=False) # 数值编码，序列化时也会带上
    # meta: Optional[Dict[str, Any]] = None  # 选填：比如来源、时间窗口等

    def __post_init__(self):
        self.type_code = encode_edge_type(self.type)

# -----------------------------
# Utilities
# -----------------------------
def wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return a

def quat_to_yaw(qw, qx, qy, qz) -> float:
    """Extract yaw (rotation around vertical axis) from quaternion in Habitat convention (w, x, y, z)."""
    siny_cosp = 2.0 * (qw * qy + qx * qz)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def rel_pose(pu: Dict[str, float], pv: Dict[str, float]) -> Tuple[float, float, float]:
    """Relative pose pv w.r.t. pu in x-z plane + yaw."""
    dx = pv["x"] - pu["x"]
    dz = pv["z"] - pu["z"]
    dyaw = wrap_to_pi(pv["yaw"] - pu["yaw"])
    return dx, dz, dyaw

def imread_rgb_uint8(path: str) -> np.ndarray:
    import cv2, os
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or BGRA or Gray
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:  # Gray -> RGB
        img = np.stack([img]*3, axis=-1)
    if img.shape[2] == 4:  # BGRA -> BGR
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)

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

def to_np_depth_m(depth) -> np.ndarray:
    """
    把深度统一成 numpy float32 米；支持 torch / numpy，支持 HxW 或 HxWx1。
    """
    if depth is None:
        return None
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    return depth
    


def _cluster_1d_dbscan(y: np.ndarray, eps: float, min_samples: int):
    if y.size == 0:
        return [], [], [], []
    labels = DBSCAN(eps=eps, min_samples=max(1, int(min_samples))).fit(y.reshape(-1,1)).labels_
    centers, mins, maxs, counts = [], [], [], []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue
        ys = y[labels == lab]
        centers.append(float(ys.mean()))
        mins.append(float(ys.min()))
        maxs.append(float(ys.max()))
        counts.append(int(len(ys)))
    if not centers:
        return [], [], [], []
    order = np.argsort(centers)
    return (np.asarray(centers)[order].tolist(),
            np.asarray(mins)[order].tolist(),
            np.asarray(maxs)[order].tolist(),
            np.asarray(counts)[order].tolist())

def _ranges_from_centers_and_extrema(centers, mins, maxs, tiny_margin: float = 0.05):
    F = len(centers)
    if F == 0:
        return []
    if F == 1:
        return [(mins[0] - tiny_margin, maxs[0] + tiny_margin)]
    mids = [(centers[i] + centers[i+1]) * 0.5 for i in range(F-1)]
    ranges = []
    ranges.append((mins[0] - tiny_margin, mids[0] + tiny_margin))
    for i in range(1, F-1):
        ranges.append((mids[i-1] - tiny_margin, mids[i] + tiny_margin))
    ranges.append((mids[-1] - tiny_margin, maxs[-1] + tiny_margin))
    return ranges

# 新版：动态 margin
def _ranges_from_centers_and_extrema_dyn(centers: List[float], mins: List[float], maxs: List[float],
                                         base_margin: float = 0.05, frac: float = 0.15):
    F = len(centers)
    if F == 0:
        return []
    if F == 1:
        return [(mins[0] - base_margin, maxs[0] + base_margin)]

    centers = np.asarray(centers, dtype=np.float32)
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)

    mids = (centers[:-1] + centers[1:]) * 0.5
    def dyn(i):
        left_gap  = centers[i] - centers[i-1] if i-1 >= 0   else np.inf
        right_gap = centers[i+1] - centers[i] if i+1 < F    else np.inf
        gap = float(min(left_gap, right_gap))
        return max(base_margin, frac * gap) if np.isfinite(gap) else base_margin

    ranges = []
    m0 = dyn(0)
    ranges.append((float(mins[0] - m0), float(mids[0] + m0)))
    for i in range(1, F-1):
        mi = dyn(i)
        ranges.append((float(mids[i-1] - mi), float(mids[i] + mi)))
    mL = dyn(F-1)
    ranges.append((float(mids[-1] - mL), float(maxs[-1] + mL)))
    return ranges

def _angdiff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)

def _ensure_uint8(rgb) -> np.ndarray:
    if rgb.dtype == np.uint8:
        arr = rgb
    else:
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr

# 并查集
class DSU:
    def __init__(self, n:int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x:int)->int:
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self, a:int, b:int):
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return
        if self.r[ra]<self.r[rb]: ra, rb = rb, ra
        self.p[rb]=ra
        if self.r[ra]==self.r[rb]: self.r[ra]+=1

# -----------------------------
# RGB/Depth Observation Saver
# -----------------------------
class ObservationSaver:
    """
    Configurable saver for RGB/Depth observations.
    - RGB can be saved as jpg or png at stride or 'all'
    - Depth can be saved as 16-bit PNG (with scale) or npy at stride or 'all'
    - Writes a frames_meta.jsonl mapping frame_id -> file paths, pose, yaw
    """
    def __init__(self, root_dir: str,
                 save_rgb: Literal["none", "stride", "all"] = "none",
                 rgb_stride: int = 30,
                 rgb_format: Literal["jpg", "png"] = "jpg",
                 save_depth: Literal["none", "stride", "all"] = "none",
                 depth_stride: int = 30,
                 depth_format: Literal["png16", "npy"] = "png16",
                 depth_scale: float = 1000.0  # meters -> millimeters
                 ):
        self.root = root_dir
        self.obs_dir = os.path.join(self.root, "obs")
        self.rgb_dir = os.path.join(self.obs_dir, "rgb")
        self.depth_dir = os.path.join(self.obs_dir, "depth")
        os.makedirs(self.obs_dir, exist_ok=True)
        if save_rgb != "none":
            os.makedirs(self.rgb_dir, exist_ok=True)
        if save_depth != "none":
            os.makedirs(self.depth_dir, exist_ok=True)

        self.save_rgb = save_rgb
        self.rgb_stride = max(1, int(rgb_stride))
        self.rgb_format = rgb_format

        self.save_depth = save_depth
        self.depth_stride = max(1, int(depth_stride))
        self.depth_format = depth_format
        self.depth_scale = float(depth_scale)

        self.meta_f = open(os.path.join(self.obs_dir, "frames_meta.jsonl"), "w")

        # write a small meta.json for depth encoding info
        with open(os.path.join(self.obs_dir, "meta.json"), "w") as f:
            json.dump({"depth_format": depth_format, "depth_scale": self.depth_scale}, f, indent=2)

    def _save_rgb(self, frame_id: int, rgb: np.ndarray) -> Optional[str]:
        import cv2
        fn = os.path.join(self.rgb_dir, f"{frame_id:06d}.{self.rgb_format}")
        if self.rgb_format == "jpg":
            cv2.imwrite(fn, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            cv2.imwrite(fn, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return fn

    def _save_depth_png16(self, frame_id: int, depth_m: np.ndarray) -> Optional[str]:
        import cv2
        # Encode depth (meters) to 16-bit PNG via scale (e.g., 1000 => millimeters).
        scaled = np.clip(depth_m * self.depth_scale, 0, 65535).astype(np.uint16)
        fn = os.path.join(self.depth_dir, f"{frame_id:06d}.png")
        cv2.imwrite(fn, scaled)
        return fn

    def _save_depth_npy(self, frame_id: int, depth_m: np.ndarray) -> Optional[str]:
        fn = os.path.join(self.depth_dir, f"{frame_id:06d}.npy")
        np.save(fn, depth_m.astype(np.float32))
        return fn

    def maybe_save(self, frame_id: int, rgb: np.ndarray, depth: Optional[np.ndarray],
                   pose: Dict[str, float], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[str]]:
        rgb_path, depth_path = None, None

        # RGB policy
        if self.save_rgb == "all" or (self.save_rgb == "stride" and frame_id % self.rgb_stride == 0):
            rgb_path = self._save_rgb(frame_id, rgb)

        # Depth policy
        if depth is not None and (self.save_depth == "all" or (self.save_depth == "stride" and frame_id % self.depth_stride == 0)):
            if self.depth_format == "png16":
                depth_path = self._save_depth_png16(frame_id, depth)
            else:
                depth_path = self._save_depth_npy(frame_id, depth)

        # write jsonl meta
        rec = {"frame_id": int(frame_id), "rgb_path": rgb_path, "depth_path": depth_path,
               "pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"], "yaw": pose["yaw"], 
               "quat": list(map(float, pose.get("quat", [1.0, 0.0, 0.0, 0.0])))}}
        
        if extra:
            rec.update(extra)
        self.meta_f.write(json.dumps(rec) + "\n")
        return {"rgb_path": rgb_path, "depth_path": depth_path}

    def close(self):
        try:
            self.meta_f.close()
        except Exception:
            pass

# -----------------------------
# Semantic (GroundedSAM2) Saver
# -----------------------------
class SemanticSaver:
    """
    逐帧保存 GroundedSAM2 的检测结果到 JSONL：
    每行 = {
      "frame_id": int,
      "image_wh": [w,h],
      "class_names": [...],
      "input_boxes": [[x1,y1,x2,y2], ...],
      "masks": [ { "size":[H,W], "counts":"..." }, ... ],
    }
    """
    def __init__(self, root_dir: str):
        self.sem_dir = os.path.join(root_dir, "obs", "sem")
        os.makedirs(self.sem_dir, exist_ok=True)
        self.f_jsonl = open(os.path.join(self.sem_dir, "frames_sem.jsonl"), "w")

    def write(self, frame_id: int, det: Dict, image_wh: Tuple[int, int]):
        boxes = det.get("input_boxes", [])
        if isinstance(boxes, np.ndarray):
            input_boxes = boxes.tolist() if boxes.size > 0 else []
        else:
            # 已经是 [] 或 list
            input_boxes = boxes or []
        
        rec = {
            "frame_id": int(frame_id),
            "image_wh": [int(image_wh[0]), int(image_wh[1])],
            "class_names": det.get("class_names", []),
            "input_boxes": input_boxes,
            "masks": det.get("masks", []),            # RLE 字典数组，已可 JSON 序列化
        }
        self.f_jsonl.write(json.dumps(rec) + "\n")

    def close(self):
        try:
            self.f_jsonl.close()
        except Exception:
            pass
# -----------------------------
# DINOv2 Global Encoder
# -----------------------------
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


#################
##### utils for 提取每个frame中的instance dinov2 feature
##### 这些feature通过patch token feature来得到
#################

# ============ 新增：RLE -> mask ============
def _rle_to_bool_mask(rle: dict) -> np.ndarray:
    m = mask_util.decode(rle)  # (H,W,1) 或 (H,W)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(bool)

def _downweight_mask_to_grid(mask_bool: np.ndarray, Hp: int, Wp: int) -> torch.Tensor:
    """
    mask(H,W){0,1} -> 软权重网格(Hp,Wp) in [0,1], 用于对 patch-tokens 做加权池化。
    """
    m = torch.from_numpy(mask_bool.astype(np.float32))  # [H,W]
    m = torch.nn.functional.interpolate(
        m[None, None, :, :], size=(Hp, Wp), mode="bilinear", align_corners=False
    )[0, 0]  # [Hp, Wp]
    return m

def _extract_tokens_and_cls(dinov2_encoder, rgb_uint8: np.ndarray):
    """
    返回:
      tokens_hw: torch.Tensor [Hp, Wp, D]  （重排为2D网格）
      cls_tok:   torch.Tensor [D]
    """
    dinov2_encoder.model.eval()
    with torch.no_grad():
        x = dinov2_encoder.prep(_ensure_uint8(rgb_uint8)).unsqueeze(0).to(dinov2_encoder.device)  # [1,3,H,W]
        out = dinov2_encoder.model.get_intermediate_layers(
            x, n=1, return_class_token=True
        )[0]  # [1, 1+N, D]
        patch_tokens, cls_token = out
        cls_tok = cls_token[0]         # [D]
        toks = patch_tokens[0]              # [N,D]

    # 自动推网格尺寸（不依赖模型配置）
    N, D = toks.shape
    w = int(round(N ** 0.5))
    h = max(1, int(round(N / max(1, w))))
    while h * w < N:
        w += 1
    toks = toks[: h * w, :].reshape(h, w, D)  # [Hp,Wp,D]
    return toks, cls_tok


def normalize_sem_to_object_list(class_names: List[Any], masks: List[Dict[str, Any]]):
    """
    把 {"class_names":[...], "masks":[RLE,...]} 规范成
    [{"class_name": str, "segmentation": RLE_dict}, ...]
    """
    obj_list = []
    if class_names is None or masks is None:
        return obj_list
    n = min(len(class_names), len(masks))
    for i in range(n):
        cname = str(class_names[i]).strip().lower()
        rle = masks[i]
        if not cname or not isinstance(rle, dict):
            continue
        if "size" in rle and "counts" in rle:
            # pycocotools 可直接 decode 这种 RLE
            obj_list.append({"class_name": cname, "segmentation": rle})
    return obj_list

def _crop_mask_region(rgb_uint8: np.ndarray, mask_bool: np.ndarray,
                      pad: int = 12) -> np.ndarray:
    """
    取 mask 的紧致包围框，外扩 pad；框内将非 mask 像素用该对象像素均值填充，得到对象裁剪图。
    用于提取instance对应的CLS Feature
    """
    yx = np.argwhere(mask_bool)
    if yx.size == 0:
        return rgb_uint8.copy()
    ys, xs = yx[:, 0], yx[:, 1]
    H, W = rgb_uint8.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    crop = rgb_uint8[y0:y1, x0:x1].copy()
    m = mask_bool[y0:y1, x0:x1]
    if m.any():
        fill = crop[m].reshape(-1, 3).mean(0).astype(np.uint8)
        crop[~m] = fill
    else:
        crop[:] = crop.mean((0, 1), dtype=np.float32).astype(np.uint8)
    return crop

def build_frame_object_features_once(
    dinov2_encoder,
    rgb_uint8: np.ndarray,
    object_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    输入：
      - dinov2_encoder: 你的 DinoGlobalEncoder 实例
      - rgb_uint8: H×W×3 uint8 RGB
      - object_list: [{"class_name": str, "segmentation": RLE_dict}, ...]
    输出：
      - (pool_feats, cls_feats)
        pool_feats: {class_name: [feat_pool, ...]}    # mask 加权池化后的 patch 特征
        cls_feats : {class_name: [feat_cls,  ...]}    # 对象裁剪图的 CLS（对象级 CLS）
    说明：
      - 同一帧内所有对象的 CLS 向量相同；为便于下游使用，这里按对象重复附加。
    """
    rgb_uint8 = _ensure_uint8(rgb_uint8)
    pool_feats: Dict[str, List[np.ndarray]] = {}
    cls_feats : Dict[str, List[np.ndarray]] = {}

    # 先一次性提取 tokens & cls，避免对每个对象重复前向
    tokens_hw, cls_tok_global = _extract_tokens_and_cls(dinov2_encoder, rgb_uint8)  # [Hp,Wp,D], [D]
    Hp, Wp, D = tokens_hw.shape

    for obj in object_list or []:
        cname = str(obj.get("class_name", "")).strip().lower()
        rle = obj.get("segmentation", None)
        if not cname or rle is None:
            continue

        # 1) instance patch feature
        # 下采样 mask -> 网格，做加权池化
        mask = _rle_to_bool_mask(rle)
        m = _downweight_mask_to_grid(mask, Hp, Wp)  # [Hp,Wp] in [0,1]
        wgt = (m > 0.2).float() * m
        wgt = wgt / (wgt.sum() + 1e-6)
        wgt = wgt.to(tokens_hw.device)

        feat_pool = (tokens_hw * wgt[..., None]).sum(dim=(0, 1))  # [D]
        feat_pool = (feat_pool / (feat_pool.norm() + 1e-6)).cpu().numpy().astype(np.float32)

        pool_feats.setdefault(cname, []).append(feat_pool)

        # 2) instance CLS: 对“对象裁剪图”前向一次，取其 CLS
        crop_rgb = _crop_mask_region(rgb_uint8, mask, pad=12)
        _, cls_tok = _extract_tokens_and_cls(dinov2_encoder, crop_rgb)
        f_cls_obj = (cls_tok / (cls_tok.norm() + 1e-6)).cpu().numpy().astype(np.float32)
        cls_feats.setdefault(cname, []).append(f_cls_obj)

    return pool_feats, cls_feats

def _flatten_class_detections_in_node(
    node_frames: List[int],
    class_name: str,
    sem_dets_by_frame: Dict[int, dict],   # fid -> {"class_names":[...], "masks":[...]}  // 这里只用来判断有无该类
    feats_by_frame: Dict[int, dict],       # fid -> {"feat":{cls:[vec...]}, "cls":{cls:[vec...]}}  (或 "objcls")
):
    """
    返回：list of dicts:
      items[k] = {
        "fid": int,
        "det_idx": int,             # 在该帧该类里的索引（与 feats_by_frame[fid]["feat"][class_name] 对齐）
        "f_pool": np.ndarray[D],    # 对象的 mask-pooled 特征
        "f_cls":  np.ndarray[D2] or None,  # 对象级 CLS（若未计算则为 None）
      }
    """
    out = []
    for fid in node_frames:
        # 语义存在即可（用来确认该帧有此类；不强依赖其顺序）
        rec  = sem_dets_by_frame.get(fid)
        maps = feats_by_frame.get(fid)
        if not rec or not maps:
            continue

        # 取对象级特征：pool & cls（cls 可能存储在 "cls" 或 "objcls"）
        pool_list = (maps.get("feat",   {}) or {}).get(class_name, [])
        cls_list  = (maps.get("cls",    {}) or {}).get(class_name, [])
        if not cls_list:
            cls_list = (maps.get("objcls", {}) or {}).get(class_name, [])

        for det_idx, fpool in enumerate(pool_list):
            f_cls = None
            if det_idx < len(cls_list):
                f_cls = np.asarray(cls_list[det_idx], dtype=np.float32)
            out.append({
                "fid": int(fid),
                "det_idx": int(det_idx),
                "f_pool": np.asarray(fpool, dtype=np.float32),
                "f_cls":  f_cls,
            })
    return out

def associate_instances_node_class(
    node_frames: List[int],
    class_name: str,
    sem_dets_by_frame: Dict[int, dict],
    feats_by_frame: Dict[int, dict],
    world_centroid_by_ref = None,       # (fid, det_idx) -> [x,y,z] or None
    sim_thr: float = 0.25,              # 使用“余弦距离”阈值；等价于 cos_sim >= 1 - sim_thr
    w_pool: float = 1.0,                # pool 相似度权重
    w_cls:  float = 0.3,                # 对象级 CLS 相似度权重
    dist3d_eps: float = 0.7,            # 3D 距离阈值（米）
) -> List[List[Tuple[int,int]]]:
    """
    返回：groups: List[List[(frame_id, det_idx)]]
      仅用对象级特征做关联：s = w_pool*cos(f_pool_i,f_pool_j) + w_cls*cos(f_cls_i,f_cls_j)
      若任一项缺失，则退化为可用项；可选加入 3D 距离门限。
    """
    items = _flatten_class_detections_in_node(
        node_frames=node_frames, class_name=class_name,
        sem_dets_by_frame=sem_dets_by_frame,
        feats_by_frame=feats_by_frame
    )
    n = len(items)
    if n == 0:
        return []

    # 预归一化特征
    Fp = np.stack([it["f_pool"] for it in items], 0).astype(np.float32)
    Fp /= (np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-6)

    has_cls = all(it["f_cls"] is not None for it in items)
    if has_cls and w_cls > 1e-9:
        Fc = np.stack([it["f_cls"] for it in items], 0).astype(np.float32)
        Fc /= (np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-6)
    else:
        Fc = None
        w_cls = 0.0  # 统一禁用 cls 分支

    # 3D 质心（可选）
    P = None
    if world_centroid_by_ref is not None:
        P = []
        for it in items:
            p = world_centroid_by_ref(it["fid"], it["det_idx"])
            P.append(None if p is None else np.asarray(p, dtype=np.float32))
        if all(p is None for p in P):
            P = None

    # 计算相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    Sp = cosine_similarity(Fp)                # [n,n]
    Scls = cosine_similarity(Fc) if Fc is not None else None

    s_thr = 1.0 - float(sim_thr)

    # 并查集合并
    dsu = DSU(n)
    for i in range(n):
        for j in range(i+1, n):
            s = w_pool * Sp[i, j]
            if Scls is not None:
                s += w_cls * Scls[i, j]
            if s < s_thr:
                continue
            if P is not None and (P[i] is not None) and (P[j] is not None):
                if float(np.linalg.norm(P[i] - P[j])) > dist3d_eps:
                    continue
            dsu.union(i, j)

    # 输出连通分量
    comp = {}
    for i in range(n):
        r = dsu.find(i)
        comp.setdefault(r, []).append(i)
    groups = [[(items[k]["fid"], items[k]["det_idx"]) for k in ids] for ids in comp.values()]
    return groups


# ----------------------------
# Utils for temporal edge constrution
# ----------------------------
from collections import defaultdict, Counter

def load_traj_groups(meta_jsonl_path: str) -> Dict[Tuple[int,int], List[int]]:
    """
    返回：groups[(episode_id, subgoal_id)] = [frame_id...]，按 traj_step 严格升序（相同 step 保留记录顺序）
    """
    tmp = defaultdict(list)  # (ep,sg) -> [(step,fid,seq)]
    with open(meta_jsonl_path, "r") as f:
        seq = 0
        for ln in f:
            rec = json.loads(ln)
            fid = int(rec["frame_id"])
            ep  = int(rec.get("episode_id", 0))
            sg  = int(rec.get("subgoal_id", -1))
            st  = int(rec.get("traj_step" , -1))
            if sg < 0 or st < 0: 
                continue
            tmp[(ep,sg)].append((st, seq, fid))
            seq += 1
    groups = {}
    for k, arr in tmp.items():
        arr.sort(key=lambda x: (x[0], x[1]))  # 先按 traj_step，再按写入顺序保证稳定
        groups[k] = [fid for (_,_,fid) in arr]
    return groups

def dedup_consecutive(seq: List[int]) -> List[int]:
    out = []
    for x in seq:
        if not out or x != out[-1]:
            out.append(x)
    return out

def build_temporal_edges_strict(
    traj_groups: Dict[Tuple[int,int], List[int]],
    frame_id_to_node: Dict[int, int],
    nodes_center_xz: np.ndarray    # [M,2]
) -> List[Edge]:
    """
    对每条 (ep,sg) 轨迹：
      1) 按 traj_step 排序后的帧 → 映射到节点
      2) 只做相邻去重（不做 tau 压缩）
      3) 依次连边（A→B→E→C→D...），统计 count 次数
    """
    import math
    from collections import Counter

    Ecnt = Counter()  # (u,v) -> count
    for key, fids in traj_groups.items():
        nodes_seq = []
        for fid in fids:
            nid = frame_id_to_node.get(int(fid), -1)
            if nid >= 0:
                nodes_seq.append(nid)
        # 相邻去重
        dedup = []
        for n in nodes_seq:
            if not dedup or n != dedup[-1]:
                dedup.append(n)
        # 统计转移
        for u, v in zip(dedup[:-1], dedup[1:]):
            if u != v:
                Ecnt[(u, v)] += 1

    # 实例化 Edge
    edges: List[Edge] = []
    for (u, v), cnt in Ecnt.items():
        pu, pv = nodes_center_xz[u], nodes_center_xz[v]
        dx, dz = float(pv[0]-pu[0]), float(pv[1]-pu[1])
        dist = float(math.hypot(dx, dz))
        edges.append(Edge(
            source_id=int(u),
            target_id=int(v),
            type="temporal",
            delta_pose={"dx": dx, "dz": dz, "dyaw": 0.0},
            distance=dist,
            count=int(cnt),
        ))
    return edges



# -----------------------------
# Core builder
# -----------------------------
class PlaceGraphBuilder:
    """Free-explore the scene (random same-island subgoals + 360° scan at subgoals)
    and build a coarse place-graph from DINOv2 features.
    """
    def __init__(self, args, memory_path: Optional[str] = None,
                 preload_dino=None, 
                 preload_gsam=None,
                 env=None
                 ):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.Env = env

        # DINOv2 encoder
        if preload_dino is None:
            dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
            self.encoder = DinoGlobalEncoder(pre_load_dinov2=dinov2, device=self.device)
        else:
            self.encoder = DinoGlobalEncoder(pre_load_dinov2=preload_dino, device=self.device)

        # Grounded SAM2
        self._gsam2 = preload_gsam
        self.obj_text_prompt = args.predefined_class
        
        self.rgb_paths: List[Optional[str]] = []

        # Where to save outputs
        if memory_path:
            self.save_dir = memory_path
        else:
            base = getattr(args, "memory_path", "./memory_out")
            scene = getattr(args, "scene_name", "scene")
            self.save_dir = os.path.join(base, scene)
        os.makedirs(self.save_dir, exist_ok=True)

        # Buffers for exploration log
        self.frames: List[int] = []
        self.poses_xyz: List[List[float]] = []
        self.yaws: List[float] = []
        self.features: List[np.ndarray] = []
        self.quats = [] # (rot.wxyz)
        self.frame_id = 0
        self.base_height = [] # 用于存放样本的每个point的高度，用来后续的楼层的操作

        # Saver not created until explore() is called (so user can choose policy)
        self.obs_saver: Optional[ObservationSaver] = None
        # self.sem_saver: Optional[SemanticSaver] = None 

        # Visualization for exploration trajectory
        self._td_base = None      # 顶视“灰度”底图，只在内存中
        self._td_height = None    # 底图对应的高度切片
        self._visited_map = None  # 与底图同 shape 的 bool 覆盖图

        # Frontier explore variables
        self.gs = args.gs
        self.cs = args.cs
        self.depth_sample_rate = self.args.depth_sample_rate

        self.calib_mat = get_sim_cam_mat_with_fov(self.args.height, self.args.width, fov=90) # Rotation matrix
        self.min_depth = self.args.min_depth
        self.max_depth = self.args.max_depth
        
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.FrontierMap = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.max_height = np.full((self.gs, self.gs), -np.inf)

        self.floor_height = self.args.floor_height
        self.map_height = self.args.map_height
        
        self.maxh = int(self.map_height / self.cs) 
        self.minh = int(self.floor_height / self.cs)

        self.inv_init_base_tf = []

        self.base_transform = np.eye(4)
        self.base_transform[0, :3] = self.args.base_forward_axis
        self.base_transform[1, :3] = self.args.base_left_axis
        self.base_transform[2, :3] = self.args.base_up_axis
        
        self.base2cam_tf = np.eye(4)
        self.base2cam_tf[:3, :3] = np.array([self.args.base2cam_rot]).reshape((3, 3))
        self.base2cam_tf[1, 3] = self.args.sensor_height

    # ---------- Exploration helpers ----------
    def _record_step(self, rgb_in: np.ndarray, depth_in: Optional[np.ndarray], agent_state,
                     jpg_stride: Optional[int] = None, meta: Optional[Dict[str, Any]] = None):
        """Compute DINO feature, pose, append to buffers, and maybe save RGB/Depth."""

        rgb_uint8 = to_np_rgb3(rgb_in)
        depth   = to_np_depth_m(depth_in)

        # Pose
        pos = agent_state.position
        rot = agent_state.rotation
        yaw = quat_to_yaw(rot.w, rot.x, rot.y, rot.z)
        pose = {"x": float(pos[0]), 
                "y": float(pos[1]), 
                "z": float(pos[2]), 
                "yaw": float(yaw),
                "quat":[float(rot.w), float(rot.x), float(rot.y), float(rot.z)]}

        # Feature
        feat = self.encoder.encode(rgb_uint8)  # [D]
        self.frames.append(int(self.frame_id))
        self.poses_xyz.append([pose["x"], pose["y"], pose["z"]])
        self.yaws.append(pose["yaw"])
        self.quats.append(pose["quat"])
        self.features.append(feat.astype(np.float32))

        # Save observations according to policy
        if self.obs_saver is not None:
            self.obs_saver.maybe_save(self.frame_id, rgb_uint8, depth, pose, extra=meta)

        """
        不在explore中对其进行语意理解，可以后续集体跑
        """
        # if self._gsam2 is not None and self.sem_saver is not None:
        #     try:
        #         prompt = self.obj_text_prompt
        #         det = self._gsam2.detect(
        #             rgb=rgb_uint8,
        #             text_prompt=prompt
        #         )
        #         H, W = rgb_uint8.shape[:2]
        #         self.sem_saver.write(self.frame_id, det, image_wh=(W, H))
        #         # --- 可选：遇到空检测也记一条轻量日志 ---
        #         if len(det["class_names"]) == 0:
        #             print(f"[INFO] GSAM no-detect at frame {self.frame_id}")
        #     except Exception as e:
        #         import traceback
        #         print(f"[WARN] GSAM detect failed at frame {self.frame_id}: {e.__class__.__name__}: {e}")
        #         traceback.print_exc(limit=1)   # 打印一行栈顶，别太吵

        self.frame_id += 1

    def _save_npz(self, fname="explore_log.npz"):
        """Save minimal exploration log for offline graph construction."""
        feats = np.stack(self.features, 0).astype(np.float16)  # [N,D]
        np.savez_compressed(os.path.join(self.save_dir, fname),
                            frame_ids=np.asarray(self.frames, dtype=np.int32),
                            poses_xyz=np.asarray(self.poses_xyz, dtype=np.float32),
                            yaws=np.asarray(self.yaws, dtype=np.float32),
                            feats=feats,
                            quats=np.asarray(self.quats, dtype=np.float32))

    def _print_islands_once(self, pf, samples: int = 5000):
        """
        随机采样可导航点，统计 scene 中的 island 连通块数量并打印。
        注意：get_random_navigable_point() 已经是可导航点，一般无需再 is_navigable 检查。
        """
        ids = set()
        for _ in range(samples):
            p = pf.get_random_navigable_point()
            iid = int(pf.get_island(p))
            if iid >= 0:
                ids.add(iid)
        ids = sorted(ids)
        scene = getattr(self.args, "scene_name", "scene")
        head = ", ".join(str(i) for i in ids[:10])
        tail = " ..." if len(ids) > 10 else ""
        print(f"[PlaceGraph] Scene '{scene}' islands = {len(ids)}  (ids: {head}{tail})")

    # ---------- Depth, Frontier helpers ----------
    def _get_sensor_spec(self, env, prefer="depth"):
        sim = getattr(env, "sims", getattr(env, "sim", None))
        agent = sim.get_agent(0)
        sensors = getattr(agent, "_sensors", {})
        if prefer in sensors:
            return sensors[prefer].specification()
        if "rgb" in sensors:
            return sensors["rgb"].specification()
        # 最后兜底：从已有传感器里捞一个
        for _, s in sensors.items():
            return s.specification()
        raise RuntimeError("No camera sensor found on agent.")

    def _compute_K_from_spec(self, spec):
        # spec.resolution = [H, W]；Habitat pinhole 相机
        H, W = int(spec.resolution[0]), int(spec.resolution[1])
        hfov_deg = float(getattr(spec, "hfov", 90.0))  # 若未设置，默认 90°
        hfov = np.deg2rad(hfov_deg)
        fx = (W * 0.5) / np.tan(hfov * 0.5)
        # 用宽高比推 vfov
        fy = fx * (H / float(W))
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        return K, (H, W)

    def _quat_wxyz_to_R(self, w, x, y, z):
        # 单位四元数 -> 3x3 旋转矩阵（右手，Y向上）
        # 兼容 Habitat (w,x,y,z)
        n = w*w + x*x + y*y + z*z
        if n < 1e-12:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        wx, wy, wz = s*w*x, s*w*y, s*w*z
        xx, xy, xz = s*x*x, s*x*y, s*x*z
        yy, yz, zz = s*y*y, s*y*z, s*z*z
        R = np.array([
            [1 - (yy + zz),     xy - wz,        xz + wy],
            [xy + wz,           1 - (xx + zz),  yz - wx],
            [xz - wy,           yz + wx,        1 - (xx + yy)]
        ], dtype=np.float32)
        return R

    def _yaw_to_R(self, yaw_rad: float):
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        # 绕世界 Y 轴旋转（右手）
        return np.array([[ c, 0, s],
                        [ 0, 1, 0],
                        [-s, 0, c]], dtype=np.float32)

    def _camera_world_pose_from_agent(self, agent_pos: np.ndarray, agent_rot_wxyz: tuple,
                                    sensor_spec) -> tuple:
        """
        返回 (R_wc, t_wc)
        - agent_rot_wxyz: (w,x,y,z)；若为 None，则仅用 yaw 构造
        - 传感器相对 agent 的外参：用 spec.position / spec.orientation
        """
        # agent 姿态
        if agent_rot_wxyz is not None:
            R_wa = self._quat_wxyz_to_R(*agent_rot_wxyz)
        else:
            # 如果 frames_meta 只有 yaw（弧度），可改用 _yaw_to_R
            raise RuntimeError("agent_rot_wxyz is required or adapt to use yaw-only")

        t_wa = agent_pos.reshape(3,)

        # 传感器相对 agent
        offs = np.array(sensor_spec.position, dtype=np.float32).reshape(3,)
        # 若相机有俯仰等，也可用 sensor_spec.orientation（三个欧拉角，弧度），这里通常是 [0,0,0]
        R_as = np.eye(3, dtype=np.float32)

        # 相机世界：R_ws = R_wa * R_as；t_ws = R_wa*offs + t_wa
        R_ws = R_wa @ R_as
        t_ws = R_wa @ offs + t_wa
        return R_ws, t_ws

    def _to_grid_no_rotate(self, env, x_world: float, z_world: float, td_map_2d: np.ndarray):
        """
        与 colorize_topdown_map 一致的网格映射（不做旋转补偿）
        """
        H0, W0 = td_map_2d.shape[:2]
        sim = getattr(env, "sims", getattr(env, "_sim", None))
        r, c = to_grid(
            realworld_x=float(z_world),   # 注意：realworld_x ← world.z
            realworld_y=float(x_world),   #       realworld_y ← world.x
            grid_resolution=(H0, W0),
            pathfinder=sim.pathfinder,
        )
        return int(r), int(c)

    # ---------- Public API: Explore ----------
    def explore(self, env=None, random_move_num: int = 50, turn_left_deg: float = 30.0,
                lock_floor: bool = False, floor_band_m: float = 0.30,
                # Observation saving policy:
                save_rgb: Literal["none", "stride", "all"] = "none",
                rgb_stride: int = 30,
                rgb_format: Literal["jpg", "png"] = "jpg",
                save_depth: Literal["none", "stride", "all"] = "none",
                depth_stride: int = 30,
                depth_format: Literal["png16", "npy"] = "png16",
                depth_scale: float = 1000.0):
        """Free explore and record (feature+pose), with optional RGB/Depth saving policy.

        Args:
            env: Optional existing NavEnv; if None we create one from args.
            random_move_num: number of random subgoals to visit.
            turn_left_deg: turning step for 360° scan.
            lock_floor: if True, constrain subgoals to be within +/- floor_band_m of starting height.
            floor_band_m: allowed height difference for lock_floor.

            save_rgb: "none" | "stride" | "all"
            rgb_stride: save every N frames when save_rgb=="stride"
            rgb_format: "jpg" or "png"

            save_depth: "none" | "stride" | "all"
            depth_stride: save every N frames when save_depth=="stride"
            depth_format: "png16" (16-bit PNG with meters*scale) or "npy"
            depth_scale: scale used if depth_format=="png16"
        """
        # Prepare environment
        if env is None:
            if NavEnv is None:
                raise RuntimeError("NavEnv import failed and no env provided.")
            env = NavEnv(self.args, init_state=None, build_map=False)
        

        # Create saver based on policy
        self.obs_saver = ObservationSaver(
            root_dir=self.save_dir,
            save_rgb=save_rgb, rgb_stride=rgb_stride, rgb_format=rgb_format,
            save_depth=save_depth, depth_stride=depth_stride,
            depth_format=depth_format, depth_scale=depth_scale
        )

        # self.sem_saver = SemanticSaver(self.save_dir)

        # First observation
        obs = env.sims.get_sensor_observations(0)
        agent_state = env.agent.get_state()
        if "rgb" not in obs:
            raise RuntimeError("Observation does not contain 'rgb'. Please check your sensor config.")
        rgb = obs["rgb"]
        depth = obs.get("depth", None)
        self._record_step(rgb, depth, agent_state)

        # Starting height (for lock_floor)
        y0 = agent_state.position[1]

        pf = env.plnner.pathfinder

        # 给整次explore一个explore_id，方便后续的temporal edge建立
        if not hasattr(self, "_episode_id"):
            self._episode_id = 0
        self._episode_id += 1
        episode_id = self._episode_id

        # Random subgoal exploration
        for k in range(random_move_num):
            # Sample a same-island navigable point (optionally constrained by height band)
            curr_pos = env.agent.get_state().position
            print("current position:", curr_pos)
            island_curr = pf.get_island(curr_pos)

            while True:
                subgoal = pf.get_random_navigable_point()
                if not pf.is_navigable(subgoal):
                    continue
                if pf.get_island(subgoal) != island_curr:
                    continue
                if lock_floor and abs(subgoal[1] - y0) > floor_band_m:
                    continue
                break

            # Move to point (returns a list of action strings)
            try:
                path, _goal = env.move2point(subgoal)
            except Exception as e:
                print("[WARN] move2point failed:", e)
                continue
            # 收集trajectory路径 并写meta
            traj_xyz = [np.array(env.agent.get_state().position, dtype=np.float32)]
            traj_step = 0

            # Execute path; record each step
            for act in path:
                if act == "stop":
                    continue
                obs = env.sims.step(act)
                agent_state = env.agent.get_state()
                rgb = obs["rgb"]
                depth = obs.get("depth", None)
                self._record_step(rgb, depth, agent_state, 
                            meta={
                                "episode_id": int(episode_id),
                                "subgoal_id": int(k),
                                "traj_step": int(traj_step),
                                "action": str(act)
                            })
                traj_step += 1
                traj_xyz.append(np.array(agent_state.position, dtype=np.float32))
            
            self.base_height.append(env.agent.get_state().position[1])

            # 360° scan at subgoal
            n_turns = int(round(360.0 / float(turn_left_deg)))
            for _t in range(max(1, n_turns)):
                obs = env.sims.step("turn_left")
                agent_state = env.agent.get_state()
                rgb = obs["rgb"]
                depth = obs.get("depth", None)
                self._record_step(rgb, depth, agent_state, 
                            meta={
                                "episode_id": int(episode_id),
                                "subgoal_id": int(k),
                                "traj_step": int(traj_step),
                                "action": "turn_left",
                            })
                traj_step += 1
            


        # Save exploration log and close saver
        self._save_npz("explore_log.npz")
        if self.obs_saver is not None:
            self.obs_saver.close()
        # if self.sem_saver is not None:
        #     self.sem_saver.close()
        np.save(self.save_dir+"/base_height.npy", np.array(self.base_height))


    ##########################
    #检查有多少层
    ##########################
    def discover_floors_from_heights_meters(
        self,
        base_height_array: np.ndarray,      # 以“米”为单位的高度数组 (N,)
        current_height_m: float,            # 当前 agent 的 y（米）
        eps: float = 0.40,                  # 与 BSC-Nav 同量级
        min_samples_frac: float = 0.20,     # ≈ N/5
        percent_clip: Optional[Tuple[float,float]] = (1.0, 99.0),  # 轻度去极值
        tiny_margin_m: float = 0.05         # 每层范围上下各加一点点缓冲
    ) -> Dict[str, Any]:
        y = np.asarray(base_height_array).reshape(-1).astype(np.float32)
        if y.size == 0:
            return dict(num_floors=0, centers_m=[], ranges_m=[], current_floor=None)

        if percent_clip is not None:
            lo_p, hi_p = percent_clip
            lo, hi = np.percentile(y, [lo_p, hi_p])
            y = y[(y >= lo) & (y <= hi)]
            if y.size == 0:
                return dict(num_floors=0, centers_m=[], ranges_m=[], current_floor=None)

        ms = max(1, int(round(len(y) * float(min_samples_frac))))
        # ms = min(ms, 3) 
        centers_m, mins_m, maxs_m, counts = _cluster_1d_dbscan(y, eps=eps, min_samples=ms)
        if not centers_m:
            return dict(num_floors=0, centers_m=[], ranges_m=[], current_floor=None)

        ranges_m = _ranges_from_centers_and_extrema(centers_m, mins_m, maxs_m, tiny_margin=tiny_margin_m)
        current_floor = int(np.argmin(np.abs(np.asarray(centers_m) - float(current_height_m))))

        return dict(
            num_floors=len(centers_m),
            centers_m=[float(c) for c in centers_m],
            ranges_m=[(float(a), float(b)) for (a,b) in ranges_m],
            current_floor=current_floor,
        )

     # ============================================
     # Distance graph construction
     # ============================================

     # ---------- Build Graph helpers ----------
    @staticmethod
    def _load_sem_stats(jsonl_path: str) -> dict:
        """
        读取 explore 阶段写的 frames_sem.jsonl，汇总每帧的:
        - num_objects: 目标数 = len(input_boxes)
        - num_classes: 去重类别数 = len(set(class_names))
        返回: { frame_id: {"num_objects": int, "num_classes": int} }
        """
        stats = {}  # {frame_id: {'num_objects': int, 'num_classes': int}}
        dets  = {}  # {frame_id: {'class_names': [...], 'masks': [...], 'boxes': [...], 'scores': [...]}}
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    rec = json.loads(line)
                    fid = int(rec["frame_id"])
                    classes = rec.get("class_names", []) or []
                    mks = rec.get("masks", []) or []
                    boxes = rec.get("input_boxes", []) or []
                    stats[fid] = {
                        "num_objects": int(len(boxes)),
                        "num_classes": int(len(set(classes))),
                    }
                    dets[fid] = {
                        "class_names": classes,
                        "masks": mks,             # RLE 字典数组，已可 JSON 序列化
                    }
        except FileNotFoundError:
            # 找不到语义文件就全 0（只按外观代表性）
            pass
        return stats, dets

    @staticmethod
    def load_frames_meta(meta_jsonl_path: str) -> Dict[int, Dict]:
        """
        返回: {frame_id: {"rgb_path": str|None, "depth_path": str|None, "pose": {...}}}
        """
        idx = {}
        with open(meta_jsonl_path, "r") as f:
            for ln in f:
                rec = json.loads(ln)
                idx[int(rec["frame_id"])] = rec
        return idx

    @staticmethod
    def load_sem_dets(jsonl_path: str) -> Dict[int, dict]:
        """
        读取由 SemanticSaver 写的 frames_sem.jsonl
        返回: { frame_id: {"class_names":[...], "masks":[...], "input_boxes":[...], "image_wh":[W,H]} }
        - 其中 masks 每个元素是 RLE dict: {"size":[H,W], "counts":"..."}
        - 若某帧没有 box 字段，则给空列表
        """
        out: Dict[int, dict] = {}
        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                fid = int(rec["frame_id"])
                out[fid] = {
                    "class_names": rec.get("class_names", []) or [],
                    "masks": rec.get("masks", []) or [],
                    "input_boxes": rec.get("input_boxes", []) or [],
                    "image_wh": rec.get("image_wh", None),
                }
        return out

    @staticmethod
    def fps_on_xz(xz: np.ndarray, min_dist: float, max_nodes: Optional[int] = None, seed=42) -> List[int]:
        """
        xz: [N,2] 世界坐标 (x,z)
        min_dist: 节点之间的最小欧氏距离（米），如 3.0
        返回：被选中作为“节点中心”的帧索引列表（相对于 xz 的行号）
        """
        N = xz.shape[0]
        if N == 0: return []
        rng = np.random.default_rng(seed)
        start = int(rng.integers(N))
        # picked = [int(np.random.randint(0, N))]
        picked = [start]
        d2 = np.full(N, np.inf, dtype=np.float32)
        while True:
            last = picked[-1]
            d2 = np.minimum(d2, np.sum((xz - xz[last:last+1])**2, axis=1))
            cand = int(np.argmax(d2))
            if len(picked) >= (max_nodes or 10**9):
                break
            if math.sqrt(float(d2[cand])) < min_dist:
                break
            picked.append(cand)
        return picked
    
    @staticmethod
    def assign_members(xyz: np.ndarray, centers_idx: List[int],
                    radius: float = 3.0, y_band: float = 0.6) -> List[List[int]]:
        """
        xyz: [N,3] 世界坐标
        centers_idx: 节点中心是从哪些帧挑出来的（xz 最近）
        半径 r 内 & |Δy| <= y_band 归到该节点（允许一个帧被多个节点覆盖的话可改策略）
        返回：每个节点的成员帧的“行号索引列表”（相对 xyz）
        """
        members = [[] for _ in centers_idx]
        centers = xyz[centers_idx]
        for i, p in enumerate(xyz):
            dxy = np.linalg.norm((centers[:, [0,2]] - p[[0,2]]) , axis=1)  # xz
            dy = np.abs(centers[:,1] - p[1])
            ok = np.where((dxy <= radius) & (dy <= y_band))[0]
            if len(ok) > 0:
                j = int(ok[np.argmin(dxy[ok])])  # 归到最近中心
                members[j].append(i)
        return members

    @staticmethod
    def pick_keyframes(
        feats: np.ndarray,
        yaws: np.ndarray,
        idxs: List[int],
        K: int = 4,
        yaw_min_sep_deg: float = 90.0,
        brute_force_cap: int = 48,   # idxs 数量 ≤ 32 时做精确组合搜索
        relax_schedule=(1.0, 0.75, 0.5, 0.0)  # 放宽 yaw 约束比例
    ) -> List[int]:
        """
        选出 idxs 中彼此差距最大、视角重合度低（通过 yaw 区分）、yaw 有区分的 K 个帧。
        - 目标：最大化 Σ_{i<j} cos_dist(i,j)
        - 约束：任意两帧 yaw 差 ≥ yaw_min_sep（可按 relax_schedule 逐步放宽）
        - 小规模精确搜索 + 大规模贪心近似（稳且快）
        返回：idxs 中被选中的“行号索引”（相对 feats）
        """
        n = len(idxs)
        if n <= K:
            return list(idxs)

        sub = feats[idxs]
        yaw = yaws[idxs]
        D = cosine_distances(sub, sub)  # 距离越大越不同
        yaw_min_sep = math.radians(yaw_min_sep_deg)

        # --- 辅助：计算一个组合的目标值 & yaw 约束检查 ---
        def combo_score(c, yaw_gate: float) -> float:
            # yaw 约束：所有对儿 yaw 差 >= yaw_gate
            for a, b in combinations(c, 2):
                if _angdiff(yaw[a], yaw[b]) < yaw_gate:
                    return -1.0  # 不可行
            # 目标：成对距离和
            s = 0.0
            for a, b in combinations(c, 2):
                s += float(D[a, b])
            return s

        # --- 1) 小规模：精确组合搜索 ---
        if n <= brute_force_cap:
            best_set, best_val = None, -1.0
            for alpha in relax_schedule:
                gate = yaw_min_sep * alpha
                for c in combinations(range(n), K):
                    v = combo_score(c, gate)
                    if v > best_val:
                        best_val, best_set = v, c
                if best_set is not None:
                    return [idxs[i] for i in best_set]

        # --- 2) 大规模：贪心近似（逐步放宽 yaw 约束） ---
        # 种子：选“最离群”的样本（离中心最远）
        cen = sub.mean(0, keepdims=True)
        seed = int(np.argmax(cosine_distances(sub, cen)[:, 0]))

        for alpha in relax_schedule:
            gate = yaw_min_sep * alpha
            sel = [seed]
            # 反复加入：最大化“与已选集合的成对距离和”
            while len(sel) < K:
                best_i, best_gain = None, -1.0
                for i in range(n):
                    if i in sel:
                        continue
                    # yaw 约束
                    if any(_angdiff(yaw[i], yaw[j]) < gate for j in sel):
                        continue
                    gain = float(D[i, sel].sum())   # 与已选集合的距离和
                    if gain > best_gain:
                        best_gain, best_i = gain, i
                if best_i is None:
                    break
                sel.append(best_i)
            if len(sel) == K:
                return [idxs[i] for i in sel]

        # --- 3) 兜底（若仍未凑够 K）：忽略 yaw 约束，填满 ---
        sel = [seed]
        while len(sel) < K:
            remain = [i for i in range(n) if i not in sel]
            best_i = int(max(remain, key=lambda i: float(D[i, sel].sum())))
            sel.append(best_i)
        return [idxs[i] for i in sel]

    # ---------- Build Graph Main function ----------
    def build_spatial_node_graph(self,
                                npz_path: Optional[str] = None,
                                frames_meta_jsonl: Optional[str] = None,
                                sem_jsonl: Optional[str] = None,
                                node_radius_m: float = 3.0,
                                min_node_spacing_m: float = 3.0,
                                keyframes_per_node: int = 4,
                                yaw_min_sep_deg: float = 90.0,
                                y_band: float = 0.6,
                                floor_min_y: Optional[float] = None,
                                floor_max_y: Optional[float] = None,
                                out_json: Optional[str] = None,
                                obj_feats_by_frame: Optional[dict] = None):
        """
        空间优先的 place-graph：
        1) FPS 在 (x,z) 选节点中心
        2) r 内归属成员
        3) 每节点选 3-4 keyframes
        4) 节点内对象实例跨帧关联，提取并写入 obj_feats.npy，图里只存索引与实例名
        """
        if npz_path is None:
            npz_path = os.path.join(self.save_dir, "explore_log.npz")
        if frames_meta_jsonl is None:
            frames_meta_jsonl = os.path.join(self.save_dir, "obs", "frames_meta.jsonl")
        if sem_jsonl is None:
            sem_jsonl = os.path.join(self.save_dir, "obs", "sem", "frames_sem.jsonl")
        if out_json is None:
            out_json = os.path.join(self.save_dir, "place_graph_spatial.json")

        data = np.load(npz_path, allow_pickle=True)
        feats_all = data["feats"].astype(np.float32)    # [N,D]
        xyz_all   = data["poses_xyz"].astype(np.float32)# [N,3]
        yaws_all  = data["yaws"].astype(np.float32)     # [N]
        frames_all= data["frame_ids"].astype(np.int32)

        # --- 楼层过滤（仅用某一层的 observation 建图）---
        keep = np.ones(len(frames_all), dtype=bool)
        if (floor_min_y is not None) and (floor_max_y is not None):
            y = xyz_all[:, 1]
            keep = (y >= float(floor_min_y)) & (y <= float(floor_max_y))

        feats   = feats_all[keep]
        xyz     = xyz_all[keep]
        yaws    = yaws_all[keep]
        frames  = frames_all[keep]

        
        # 1) 节点中心（FPS）
        xz = xyz[:, [0,2]]
        centers_idx_local = self.fps_on_xz(xz, min_dist=min_node_spacing_m)
        print("node center postion done!")

        # 2) 归属成员
        members_local = self.assign_members(xyz, centers_idx_local, radius=node_radius_m, y_band=y_band)
        print("node members done!")

        #### 第二次补充。增加info score，避免不好的视角（白墙，壁画等等）
        frames_meta = self.load_frames_meta(frames_meta_jsonl)           # frame_id -> paths
        sem_dets_by_frame = self.load_sem_dets(sem_jsonl) if os.path.exists(sem_jsonl) else {}

        # 3) keyframes
        nodes = []
        for nid, (cidx, idxs) in enumerate(zip(centers_idx_local, members_local)):
            if len(idxs)==0: continue
            ksel_local = self.pick_keyframes(feats, yaws, idxs, K=keyframes_per_node, yaw_min_sep_deg=yaw_min_sep_deg)

            node = PlaceNode(
                id=len(nodes),
                center={"x": float(xyz[cidx,0]), "y": float(xyz[cidx,1]), "z": float(xyz[cidx,2])},
                radius=float(node_radius_m),
                members=[int(frames[i]) for i in idxs],
                keyframes=[int(frames[i]) for i in ksel_local],
                object_list=[]
            )
            nodes.append(node)
        print("node keyframes done!")

        # 4) 语义：跨帧实例 + 特征写盘

        # 逐帧提取所有对象 crop 特征
        # feats_by_frame = {}
        # for fid, rec in sem_dets_by_frame.items():
        #     meta = frames_meta.get(fid)
        #     if not meta or not meta.get("rgb_path"): 
        #         continue
        #     rgb_path = meta.get("rgb_path")
        #     if not rgb_path or not os.path.exists(rgb_path):
        #         continue
        #     rgb_uint8 = imread_rgb_uint8(rgb_path)

        #     class_names = rec.get("class_names", [])
        #     masks = rec.get("masks", [])
        #     obj_list = normalize_sem_to_object_list(class_names, masks)
        #     feat_map, cls_map = build_frame_object_features_once(self.encoder, rgb_uint8, obj_list)
        #     feats_by_frame[fid] = {
        #         "feat": feat_map,   # {class_name: [feat_pool, ...]}
        #         "cls":  cls_map,    # {class_name: [feat_cls,  ...]}
        #     }
        # print("node object features extraction done!")

        # 读取存储的object 
        feats_by_frame = obj_feats_by_frame

        # 4.3 建立对象特征库
        obj_feats = []  # List[np.ndarray[D]]
        def push_feat(vec: np.ndarray) -> int:
            obj_feats.append(vec.astype(np.float32))
            return len(obj_feats) - 1


        # 4.4 对每个节点：按 class 关联多视角 → 形成实例，并把特征写索引
        for node in nodes:
            fids = list(set(node.members))

            # 该 node 覆盖的类别集合
            classes_in_node = set()
            for fid in fids:
                rec = sem_dets_by_frame.get(fid)
                if rec:
                    for c in rec.get("class_names", []):
                        if c: classes_in_node.add(str(c).strip().lower())

            class_counters = {}
            node_objects = []

            for cls_name in sorted(classes_in_node):
                groups = associate_instances_node_class(
                    node_frames=fids,
                    class_name=cls_name,
                    sem_dets_by_frame=sem_dets_by_frame,
                    feats_by_frame=feats_by_frame,
                    world_centroid_by_ref=None,   # 如已实现可传
                    sim_thr=0.25, w_pool=1.0, w_cls=0.3, dist3d_eps=0.7
                )
                if not groups:
                    continue

                # 把每组写成一个实例：聚合特征 -> 写入 obj_feats.npy，记录索引
                for g in groups:
                    # 收集该实例的所有 mask-pooled 特征向量
                    vecs = []
                    view_refs = []
                    for (fid, det_idx) in g:
                        # 从 feats_by_frame 中取该视角的对象向量
                        vlist = feats_by_frame[fid]["feat"].get(cls_name, [])
                        if det_idx < len(vlist):
                            vecs.append(np.asarray(vlist[det_idx], dtype=np.float32))
                            view_refs.append([int(fid), int(det_idx)])
                    if not vecs:
                        continue
                    V = np.stack(vecs, 0)
                    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-6)
                    v_mean = V.mean(0)
                    v_mean = v_mean / (np.linalg.norm(v_mean) + 1e-6)

                    feat_idx = push_feat(v_mean)   # 写入 obj_feats，返回全局索引

                    # 实例命名：table 1/2/3...
                    k = class_counters.get(cls_name, 0) + 1
                    class_counters[cls_name] = k
                    inst_name = f"{cls_name} {k}"

                    node_objects.append({
                        "instance_name": inst_name,
                        "class_name": cls_name,
                        "feature_indices": [int(feat_idx)],   # 如需多种表征可扩展为多个索引
                        "views": view_refs,                   # [(frame_id, det_idx), ...]
                    })

            node.object_list = node_objects
        print("node object features fusion done!")
                    
        # 5) 几何边：节点中心做 kNN（和你现在的 _build_geometric_edges 类似，但用 node.center）
        # —— 仅时序边 —— 
        meta_jsonl = frames_meta_jsonl  # 上面已经算好的路径
        traj_groups = load_traj_groups(meta_jsonl)

        # frame_id -> node_id（需要先有每帧的 node 归属）
        # 简单最近中心归属（或你已有 assign_frame_to_node_per_frame 的结果）
        centers_world = np.array([[n.center["x"], n.center["y"], n.center["z"]] for n in nodes], dtype=np.float32)
        centers_xz = centers_world[:, [0,2]]
        frame_id_to_node = {}
        for i in range(len(frames)):
            fid = int(frames[i])
            p = xyz[i, [0,2]]
            if len(centers_xz)==0:
                continue
            j = int(np.argmin(np.linalg.norm(centers_xz - p[None,:], axis=1)))
            frame_id_to_node[fid] = j

        edges = build_temporal_edges_strict(
            traj_groups=traj_groups,
            frame_id_to_node=frame_id_to_node,
            nodes_center_xz=centers_xz
        )
        print("node temporal edges done!")

        # 6) 写 graph.json + obj_feats.npy
        graph = {
            "nodes": [asdict(n) for n in nodes],
            "edges": [asdict(e) for e in edges],
            "meta": {
                "node_radius_m": node_radius_m,
                "min_node_spacing_m": min_node_spacing_m,
                "keyframes_per_node": keyframes_per_node,
                "npz_path": npz_path
            },
            "features": {
                "obj_feats_path": os.path.join(self.save_dir, f"obj_feats_floor{self.args.floor_idx}_min{self.args.min_dis}_radius{self.args.radius}.npy"),
                "feat_dim": int(obj_feats[0].shape[0]) if obj_feats else 0
            }
        }
        with open(out_json, "w") as f:
            json.dump(graph, f, indent=2)
        if obj_feats:
            np.save(os.path.join(self.save_dir, f"obj_feats_floor{self.args.floor_idx}_min{self.args.min_dis}_radius{self.args.radius}.npy"), np.stack(obj_feats,0).astype(np.float32))
        return out_json
    
     # ============================================
     # Spatial Node Graph Visualization
     # ============================================

     # ---------- Visualization helpers ----------
    @staticmethod
    def world_xz_to_rot_px(env, x_world: float, z_world: float, td):
        H0, W0 = td["map"].shape  # 旋转之前的二值栅格尺寸 (rows, cols)
        # to_grid 的 realworld_x 对应世界坐标 z，realworld_y 对应世界坐标 x
        sim = getattr(env, "sims", getattr(env, "_sim", None))
        assert sim is not None and hasattr(sim, "pathfinder"), "env 没有 sim/pathfinder"
        r, c = to_grid(
            realworld_x=z_world,
            realworld_y=x_world,
            grid_resolution=(H0, W0),
            pathfinder=sim.pathfinder,
        )  # r=row, c=col

        """
        Attention
        :这里不需要旋转，如果旋转的话就会出问题
        """
        # if H0 > W0:
        #     # colorize 会先对原图 np.rot90(..., 1)（逆时针 90°）
        #     # 原 (r,c) -> 旋转后 (row' = W0-1-c, col' = r)
        #     # 我们需要 (x,y) = (col', row')
        #     x_px = float(r)
        #     y_px = float(W0 - 1 - c)
        # else:
        #     # 未旋转，直接 (x,y)=(c,r)
        x_px = float(c)
        y_px = float(r)
        return (x_px, y_px)

    @staticmethod
    def _draw_dot(img, p, color, r=4, thickness=-1):
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_text(img, p, text, color=(255, 255, 255)):
        cv2.putText(img, text, (int(p[0]) + 4, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # convert 3d points to 2d topdown coordinates
    @staticmethod
    def convert_points_to_topdown(pathfinder, points_x, points_z, meters_per_pixel):
        bounds = pathfinder.get_bounds()
        # convert 3D x,z to topdown x,y
        px = (points_x - bounds[0][0]) / meters_per_pixel
        py = (points_z - bounds[0][2]) / meters_per_pixel
        return (px, py)

    # ---------- Visualization Main function ----------
    def visualize_graph_from_json(self,
                                env,
                                graph_json_path: str,
                                out_path: str,
                                show_nodes: bool = True,
                                show_edges: bool = True,
                                node_radius: int = 3,
                                edge_thickness: int = 1,
                                canvas_h: int = 1024,
                                floor_y: float = 0.0,
                                edge_types: Optional[List[str]] = None,    # 例: ["temporal","geometric"]
                                draw_node_ids: bool = False):
        """
        在 Habitat 的 topdown 底图上渲染 place_graph_basic.json
        - 兼容两种节点格式：
            nodes[i].pose = {"x","y","z","yaw"}      （basic）
            nodes[i].center = {"x","y","z"}          （spatial）
        - 兼容边是 dict 或 dataclass(asdict 后)
        """
        assert os.path.exists(graph_json_path), f"graph json not found: {graph_json_path}"
        with open(graph_json_path, "r") as f:
            G = json.load(f)

        nodes = G.get("nodes", [])
        edges = G.get("edges", [])

        # --- 1) 顶视底图 & 缩放 ---
        y_floor = float(floor_y)
        td_map = get_topdown_map(env.sims.pathfinder, height=y_floor)  # HxW uint8
        color  = colorize_topdown_map(td_map)                                                 # HxWx3 RGB

        H0, W0 = td_map.shape[:2]
        scale  = float(canvas_h) / float(H0)
        vis    = cv2.resize(color[:, :, ::-1], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  # BGR

        # top_down_map = maps.get_topdown_map(
        #     env.sims.pathfinder, height=floor_y, meters_per_pixel=0.1
        # )
        # recolor_map = np.array(
        #     [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        # )
        # top_down_map = recolor_map[top_down_map]

        def up_xy(px_xy):  # 原始像素 -> 缩放后像素
            return (px_xy[0] * scale, px_xy[1] * scale)
        
        td = {"map": td_map}

        # --- 2) 计算节点像素坐标（兼容 pose/center）---
        id2px = {}  # node_id -> (x_px, y_px)  （未缩放）
        for n in nodes:
            nid = int(n.get("id", len(id2px)))
            if "center" in n and isinstance(n["center"], dict):
                xw = float(n["center"].get("x", 0.0))
                zw = float(n["center"].get("z", 0.0))
            elif "pose" in n and isinstance(n["pose"], dict):
                xw = float(n["pose"].get("x", 0.0))
                zw = float(n["pose"].get("z", 0.0))
            else:
                # 没有世界坐标就跳过
                continue

            x_px, y_px = self.world_xz_to_rot_px(env, xw, zw, td)  # 未缩放像素坐标
        #     x_px, y_px = self.convert_points_to_topdown(
        #     env.sims.pathfinder, xw, zw, 0.1
        # )
            id2px[nid] = (x_px, y_px)

        # --- 3) 画节点 ---
        if show_nodes:
            for nid, (x_px, y_px) in id2px.items():
                X, Y = up_xy((x_px, y_px))
                # 你的点绘制函数；若没有可用 cv2.circle 代替
                if hasattr(self, "_draw_dot"):
                    self._draw_dot(vis, (X, Y), color=(180, 220, 255), r=node_radius)
                else:
                    cv2.circle(vis, (int(X), int(Y)), node_radius, (180, 220, 255), thickness=-1, lineType=cv2.LINE_AA)
                if draw_node_ids:
                    cv2.putText(vis, str(nid), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 255, 255), 1, cv2.LINE_AA)
        # if show_nodes:
        #     for nid, (x_px, y_px) in id2px.items():
        #         X, Y = (x_px, y_px)
        #         # 你的点绘制函数；若没有可用 cv2.circle 代替
        #         if hasattr(self, "_draw_dot"):
        #             self._draw_dot(top_down_map, (X, Y), color=(180, 220, 255), r=node_radius)
        #         else:
        #             cv2.circle(top_down_map, (int(X), int(Y)), node_radius, (180, 220, 255), thickness=-1, lineType=cv2.LINE_AA)
        #         if draw_node_ids:
        #             cv2.putText(top_down_map, str(nid), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX, 
        #                         0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # --- 4) 画边（可选择类型）---
        if show_edges:
            # 颜色配置
            def color_for_type(t: str):
                t = (t or "").lower()
                if t == "temporal":  # 绿色
                    return (80, 160, 80)
                if t == "geometric": # 橙色
                    return (60, 140, 230)
                return (128, 128, 180)  # 其他

            for e in edges:
                # 兼容 dict 或 dataclass(asdict 后)
                u = e.get("source_id") if isinstance(e, dict) else getattr(e, "source_id", None)
                v = e.get("target_id") if isinstance(e, dict) else getattr(e, "target_id", None)
                et = e.get("type", "temporal") if isinstance(e, dict) else getattr(e, "type", "temporal")
                cnt = e.get("count", None) if isinstance(e, dict) else getattr(e, "count", None)
                if cnt is None:
                    # 兼容老版本把计数塞在 delta_pose["count"] 的情况
                    dp = e.get("delta_pose", {}) if isinstance(e, dict) else getattr(e, "delta_pose", {}) or {}
                    cnt = dp.get("count", 1)

                if u is None or v is None: 
                    continue
                if (u not in id2px) or (v not in id2px):
                    continue
                if edge_types and (et not in edge_types):
                    continue

                (xu, yu) = id2px[u]
                (xv, yv) = id2px[v]
                (Xu, Yu) = up_xy((xu, yu))
                (Xv, Yv) = up_xy((xv, yv))

                # 线宽按出现次数对数缩放，直观一点
                t = int(max(1, edge_thickness + min(4, int(math.log1p(max(1, int(cnt)))))))
                color = color_for_type(et)

                cv2.line(vis, (int(Xu), int(Yu)), (int(Xv), int(Yv)),
                        color=color, thickness=t, lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        # cv2.imwrite(out_path, top_down_map)
        print(f"[viz-graph-json] saved -> {out_path}")
        return out_path

    def visualize_graph_from_json_with_keyframes(self,
                                env,
                                graph_json_path: str,
                                out_path: str,
                                show_nodes: bool = True,
                                show_edges: bool = True,
                                node_radius: int = 3,
                                edge_thickness: int = 1,
                                canvas_h: int = 1024,
                                floor_y: float = 0.0,
                                edge_types: Optional[List[str]] = None,    # 例: ["temporal","geometric"]
                                draw_node_ids: bool = False,
                                # -------- 新增：keyframe 可视化 --------
                                show_keyframes: bool = False,
                                draw_kf_ids: bool = False,
                                draw_kf_yaw: bool = True,
                                show_kf_fov: bool = True,
                                kf_radius: int = 2,
                                kf_arrow_len_m: float = 0.5,      # 箭头长度（米）
                                kf_fov_deg: float = 90.0,         # HFOV（度）
                                kf_fov_range_m: float = 0.5,      # FOV 扇形半径（米）
                                kf_fov_alpha: float = 0.28,       # FOV 透明度
                                kf_color: tuple = (60, 80, 240),  # BGR
                                kf_fov_color: tuple = (120, 160, 255),  # BGR（淡色）
                                yaw_mode: str = "x_sin_z_cos",     # "x_sin_z_cos" 或 "x_cos_z_neg_sin"

                                # -------- 新增：node 半径圈 --------
                                show_node_radius: bool = False,
                                node_radius_alpha: float = 0.15,
                                node_radius_fill_color: tuple = (200, 220, 255),  # BGR（淡色）
                                node_radius_edge_color: tuple = (160, 200, 240),  # BGR（描边）
                                node_radius_edge_thickness: int = 1,
                                radius_m: float = 0.0,

                                # -------- 特定位置可视化-------
                                specific_id: int = 0
                                ):
        import os, json, math
        import numpy as np
        import cv2

        assert os.path.exists(graph_json_path), f"graph json not found: {graph_json_path}"
        with open(graph_json_path, "r") as f:
            G = json.load(f)

        nodes = G.get("nodes", [])
        edges = G.get("edges", [])
        meta  = G.get("meta", {}) or {}
        npz_path_in_meta = meta.get("npz_path", None)

        # --- 1) 顶视底图 & 缩放 ---
        y_floor = float(floor_y)
        td_map = get_topdown_map(env.sims.pathfinder, height=y_floor)  # HxW uint8
        color  = colorize_topdown_map(td_map)                          # HxWx3 RGB

        H0, W0 = td_map.shape[:2]
        scale  = float(canvas_h) / float(H0)
        vis    = cv2.resize(color[:, :, ::-1], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)  # BGR

        def up_xy(px_xy):  # 原始像素 -> 缩放后像素
            return (px_xy[0] * scale, px_xy[1] * scale)

        td = {"map": td_map}

        # --- 2) 计算节点像素坐标（兼容 pose/center）---
        id2px = {}       # node_id -> (x_px, y_px) 未缩放
        id2world = {}    # node_id -> (xw, zw)
        for n in nodes:
            nid = int(n.get("id", len(id2px)))
            if "center" in n and isinstance(n["center"], dict):
                xw = float(n["center"].get("x", 0.0))
                zw = float(n["center"].get("z", 0.0))
            elif "pose" in n and isinstance(n["pose"], dict):
                xw = float(n["pose"].get("x", 0.0))
                zw = float(n["pose"].get("z", 0.0))
            else:
                continue
            x_px, y_px = self.world_xz_to_rot_px(env, xw, zw, td)  # 未缩放像素
            id2px[nid] = (x_px, y_px)
            id2world[nid] = (xw, zw)


        # ========== 新增：3) 画每个 node 的覆盖半径 ==========
        if show_node_radius:
            # 世界 (dx, dz) 米 → 像素偏移（未缩放）
            def world_meters_to_px_from(xw, zw, dx_m, dz_m):
                x2, y2 = self.world_xz_to_rot_px(env, xw + dx_m, zw + dz_m, td)
                x1, y1 = self.world_xz_to_rot_px(env, xw,        zw,        td)
                return (x2 - x1, y2 - y1)

            overlay = vis.copy()
            for n in nodes:
                nid = int(n.get("id", -1))
                if (nid not in id2px) or (nid not in id2world):
                    continue
                if radius_m == 0.0:
                    r_m = float(n.get("radius", 0.0))
                else:
                    r_m = radius_m
                if r_m <= 0:
                    continue

                (xw, zw) = id2world[nid]
                (cx_px, cy_px) = id2px[nid]
                Cx, Cy = up_xy((cx_px, cy_px))

                # 将“米”为单位的半径换算成像素半径（取 x 向和 z 向的平均，稳一点）
                dx1, dy1 = world_meters_to_px_from(xw, zw, r_m, 0.0)
                dx2, dy2 = world_meters_to_px_from(xw, zw, 0.0, r_m)
                r_px_unscaled = 0.5 * (math.hypot(dx1, dy1) + math.hypot(dx2, dy2))
                R = max(1, int(round(r_px_unscaled * scale)))

                # 先在 overlay 上画填充，再 alpha 混合到 vis
                cv2.circle(overlay, (int(Cx), int(Cy)), R, node_radius_fill_color, thickness=-1, lineType=cv2.LINE_AA)
                # 描边直接画在 vis 上，保证边清晰
                cv2.circle(vis, (int(Cx), int(Cy)), R, node_radius_edge_color, thickness=node_radius_edge_thickness, lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, node_radius_alpha, vis, 1 - node_radius_alpha, 0, dst=vis)
        # ========== 新增结束 ==========


        # --- 3) 画节点 ---
        if show_nodes:
            for nid, (x_px, y_px) in id2px.items():
                X, Y = up_xy((x_px, y_px))
                cv2.circle(vis, (int(X), int(Y)), node_radius, (180, 220, 255), thickness=-1, lineType=cv2.LINE_AA)
                if draw_node_ids:
                    cv2.putText(vis, str(nid), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # --- 4) 画边（可选择类型）---
        if show_edges:
            def color_for_type(t: str):
                t = (t or "").lower()
                if t == "temporal":  return (80, 160, 80)   # 绿
                if t == "geometric": return (60, 140, 230)  # 橙
                return (128, 128, 180)

            for e in edges:
                u = e.get("source_id") if isinstance(e, dict) else getattr(e, "source_id", None)
                v = e.get("target_id") if isinstance(e, dict) else getattr(e, "target_id", None)
                et = e.get("type", "temporal") if isinstance(e, dict) else getattr(e, "type", "temporal")
                cnt = e.get("count", None) if isinstance(e, dict) else getattr(e, "count", None)
                if cnt is None:
                    dp = e.get("delta_pose", {}) if isinstance(e, dict) else getattr(e, "delta_pose", {}) or {}
                    cnt = dp.get("count", 1)
                if u is None or v is None: 
                    continue
                if (u not in id2px) or (v not in id2px):
                    continue
                if edge_types and (et not in edge_types):
                    continue

                (xu, yu) = id2px[u]; (xv, yv) = id2px[v]
                (Xu, Yu) = up_xy((xu, yu)); (Xv, Yv) = up_xy((xv, yv))
                t = int(max(1, edge_thickness + min(4, int(math.log1p(max(1, int(cnt)))))))
                cv2.line(vis, (int(Xu), int(Yu)), (int(Xv), int(Yv)), color_for_type(et), thickness=t, lineType=cv2.LINE_AA)

        # --- 5) 画 Keyframes（点 + yaw 箭头 + FOV 扇形） ---
        if show_keyframes:
            # 5.1 载入 npz 里全量帧位姿和 yaw
            fid2_poseyaw = {}
            if npz_path_in_meta and os.path.exists(npz_path_in_meta):
                data = np.load(npz_path_in_meta, allow_pickle=True)
                frames_all = data["frame_ids"].astype(np.int64)
                xyz_all    = data["poses_xyz"].astype(np.float32)   # [N,3]
                yaws_all   = data["yaws"].astype(np.float32)        # [N]（弧度）
                for i, fid in enumerate(frames_all):
                    fid2_poseyaw[int(fid)] = (float(xyz_all[i,0]), float(xyz_all[i,2]), float(yaws_all[i]))
            else:
                print("[viz-graph-json] WARNING: meta.npz_path 不存在，无法绘制 keyframe。")

            # 5.2 帮助：把“世界米”为单位的矢量转像素坐标
            def world_meters_to_px_from(xw, zw, dx_m, dz_m):
                # 用 world->px 转换前后两点得到像素
                x2, y2 = self.world_xz_to_rot_px(env, xw + dx_m, zw + dz_m, td)
                x1, y1 = self.world_xz_to_rot_px(env, xw,        zw,        td)
                return (x2 - x1, y2 - y1)  # 未缩放像素中的位移

            # 5.3 画一个 keyframe 的方法
            def draw_one_kf(xw, zw, yaw_rad, kfid_text=None):
                # a) keyframe 点
                px, py = self.world_xz_to_rot_px(env, xw, zw, td)
                X, Y = up_xy((px, py))
                cv2.circle(vis, (int(X), int(Y)), kf_radius, kf_color, thickness=-1, lineType=cv2.LINE_AA)
                if draw_kf_ids and kfid_text is not None:
                    cv2.putText(vis, str(kfid_text), (int(X)+3, int(Y)-3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, kf_color, 1, cv2.LINE_AA)

                # b) yaw 箭头（可选）
                if draw_kf_yaw:
                    if yaw_mode == "x_sin_z_cos":
                        dx_m = math.sin(yaw_rad) * kf_arrow_len_m
                        dz_m = math.cos(yaw_rad) * kf_arrow_len_m
                    else:  # "x_cos_z_neg_sin"
                        dx_m = math.cos(yaw_rad) * kf_arrow_len_m
                        dz_m = -math.sin(yaw_rad) * kf_arrow_len_m
                    dx_px, dy_px = world_meters_to_px_from(xw, zw, dx_m, dz_m)
                    X2, Y2 = up_xy((px + dx_px, py + dy_px))
                    cv2.arrowedLine(vis, (int(X), int(Y)), (int(X2), int(Y2)), kf_color, 1, tipLength=0.18)

                # c) FOV 扇形（可选）
                if show_kf_fov:
                    half = math.radians(kf_fov_deg * 0.5)
                    # 左右边界方向
                    if yaw_mode == "x_sin_z_cos":
                        # d = (sin(theta), cos(theta))
                        dx_l = math.sin(yaw_rad + half) * kf_fov_range_m
                        dz_l = math.cos(yaw_rad + half) * kf_fov_range_m
                        dx_r = math.sin(yaw_rad - half) * kf_fov_range_m
                        dz_r = math.cos(yaw_rad - half) * kf_fov_range_m
                    else:
                        dx_l = math.cos(yaw_rad + half) * kf_fov_range_m
                        dz_l = -math.sin(yaw_rad + half) * kf_fov_range_m
                        dx_r = math.cos(yaw_rad - half) * kf_fov_range_m
                        dz_r = -math.sin(yaw_rad - half) * kf_fov_range_m
                    # 端点像素
                    dxlp, dylp = world_meters_to_px_from(xw, zw, dx_l, dz_l)
                    dxrp, dyrp = world_meters_to_px_from(xw, zw, dx_r, dz_r)
                    Xl, Yl = up_xy((px + dxlp, py + dylp))
                    Xr, Yr = up_xy((px + dxrp, py + dyrp))

                    # 画半透明扇形（三角形近似）
                    overlay = vis.copy()
                    pts = np.array([[int(X), int(Y)], [int(Xl), int(Yl)], [int(Xr), int(Yr)]], dtype=np.int32)
                    cv2.fillConvexPoly(overlay, pts, kf_fov_color)
                    cv2.addWeighted(overlay, kf_fov_alpha, vis, 1 - kf_fov_alpha, 0, dst=vis)

            # 5.4 遍历每个 node 的 keyframes
            for n in nodes:
                kfs = n.get("keyframes", [])
                if not kfs:
                    continue
                for kfid in kfs:
                    rec = fid2_poseyaw.get(int(kfid))
                    if rec is None:
                        continue
                    xw, zw, yaw = rec
                    draw_one_kf(xw, zw, yaw, kfid_text=kfid)

        # draw a specific view 
        specific_color = (150, 150, 80)  # BGR
        rec = fid2_poseyaw.get(int(specific_id))
        xw, zw, yaw = rec
        px, py = self.world_xz_to_rot_px(env, xw, zw, td)
        X, Y = up_xy((px, py))
        cv2.circle(vis, (int(X), int(Y)), kf_radius, specific_color, thickness=-1, lineType=cv2.LINE_AA)   


        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[viz-graph-json] saved -> {out_path}")
        return out_path


    def visualize_view_points(self,
                            env,
                            frames_meta_jsonl_path: str,
                            out_path: str,
                            filter_y_range: Tuple[float, float] = (None, None),
                            point_color: Tuple[int, int, int] = (0, 0, 255),  # 红色
                            point_radius: int = 5,
                            show_labels: bool = False,
                            canvas_h = 1024,
                            y_floor = 0
                            ):
        """
        在 Habitat 的 topdown 地图上标记采集的 view 点位置。
        
        参数:
            env: Habitat环境对象
            frames_meta_jsonl_path: str, jsonl文件路径，包含每帧的pose信息
            out_path: str, 输出图片路径
            filter_y_range: Tuple[float, float], height范围筛选
            point_color: Tuple[int,int,int], 标记点颜色（BGR）
            point_radius: int, 点的半径
            show_labels: bool, 是否显示点的索引编号
        """
        # 读取jsonl文件
        assert os.path.exists(frames_meta_jsonl_path), f"jsonl文件不存在: {frames_meta_jsonl_path}"
        points = []
        with open(frames_meta_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                pose = data.get("pose", {})
                x = float(pose.get("x", 0))
                y = float(pose.get("y", 0))
                z = float(pose.get("z", 0))
                # 只筛选 y
                if (filter_y_range[0] is not None and y < filter_y_range[0]) or \
                (filter_y_range[1] is not None and y > filter_y_range[1]):
                    continue
                # 存储 (x, z)
                points.append((x, z))
        
        # 获取topdown地图
        td_map = get_topdown_map(env.sims.pathfinder, height=y_floor, meters_per_pixel=0.05)
        color_map = colorize_topdown_map(td_map)
        H0, W0 = td_map.shape[:2]
        scale = float(canvas_h) / float(H0)
        vis = cv2.resize(color_map[:, :, ::-1], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # 计算像素坐标（假设已有world_xz_to_rot_px函数）
        id_points_px = []
        for idx, (xw, yw) in enumerate(points):
            px_x, px_y = self.world_xz_to_rot_px(env, xw, yw, {"map": td_map})
            id_points_px.append((px_x, px_y))
            # 画点
            X, Y = (px_x * scale, px_y * scale)
            cv2.circle(vis, (int(X), int(Y)), point_radius, point_color, thickness=-1, lineType=cv2.LINE_AA)
            if show_labels:
                cv2.putText(vis, str(idx), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # 保存结果
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[view_points] saved -> {out_path}")
        return out_path


    def visualize_view_points_traj(self,
                            env,
                            frames_meta_jsonl_path: str,
                            out_path: str,
                            filter_y_range: Tuple[float, float] = (None, None),
                            point_color: Tuple[int, int, int] = (0, 0, 255),  # 红色
                            point_radius: int = 5,
                            show_labels: bool = False,
                            canvas_h = 1024,
                            y_floor = 0,
                            # ---- 新增：仅此两项 ----
                            selected_subgoal_ids: Optional[List[int]] = None,
                            traj_thickness: int = 2
                            ):
        """
        在 Habitat 的 topdown 地图上标记采集的 view 点位置，并支持：
        - 仅绘制指定 sub_goal_id 的轨迹与点（selected_subgoal_ids）
        - 绘制轨迹（统一颜色）
        - out_path 以 .pdf 结尾时，直接保存为 PDF（不经由图片中转）
        """
        import os, json
        import numpy as np
        import cv2
        from collections import defaultdict

        # 读取jsonl文件
        assert os.path.exists(frames_meta_jsonl_path), f"jsonl文件不存在: {frames_meta_jsonl_path}"
        points = []
        traj_by_sid = defaultdict(list)   # sid -> list[(x,z)]

        # 规范化要保留的 sub_goal_id 集合（转 int，兼容字符串）
        keep_sid_set = None
        if selected_subgoal_ids:
            keep_sid_set = set()
            for v in selected_subgoal_ids:
                try:
                    keep_sid_set.add(int(v))
                except Exception:
                    pass  # 忽略无法转成 int 的值

        with open(frames_meta_jsonl_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                pose = data.get("pose", {})
                x = float(pose.get("x", 0))
                y = float(pose.get("y", 0))
                z = float(pose.get("z", 0))

                # 只筛选 y（保留原逻辑）
                if (filter_y_range[0] is not None and y < filter_y_range[0]) or \
                (filter_y_range[1] is not None and y > filter_y_range[1]):
                    continue

                # sub_goal_id 过滤（如指定）
                sid_raw = data.get("subgoal_id", None)
                try:
                    sid_int = int(sid_raw) if sid_raw is not None else None
                except Exception:
                    sid_int = None
                if keep_sid_set is not None and sid_int not in keep_sid_set:
                    continue

                # 存储 (x, z) 与轨迹序列
                points.append((x, z))
                traj_by_sid[-1 if sid_int is None else sid_int].append((x, z))
        # 获取topdown地图
        td_map = get_topdown_map(env.sims.pathfinder, height=y_floor, meters_per_pixel=0.05)
        color_map = colorize_topdown_map(td_map)  # RGB
        H0, W0 = td_map.shape[:2]
        scale = float(canvas_h) / float(H0)

        # 计算像素坐标（假设已有world_xz_to_rot_px函数）
        id_points_px = []
        for idx, (xw, yw) in enumerate(points):
            px_x, px_y = self.world_xz_to_rot_px(env, xw, yw, {"map": td_map})
            id_points_px.append((px_x, px_y))

        # =============== 分支 A：直接保存为 PDF（不先导出图片） ===============
        if out_path.lower().endswith(".pdf"):
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.utils import ImageReader
            from reportlab.lib.colors import Color, black
            from PIL import Image

            out_w_px = int(round(W0 * scale))
            out_h_px = int(round(H0 * scale))
            page_w_pt = float(out_w_px)   # 1 px ≈ 1 pt
            page_h_pt = float(out_h_px)

            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            c = rl_canvas.Canvas(out_path, pagesize=(page_w_pt, page_h_pt))
            c.setLineCap(1); c.setLineJoin(1)

            # 底图：缩放后的 topdown 作为位图嵌入 PDF
            color_map_scaled = cv2.resize(color_map, (out_w_px, out_h_px), interpolation=cv2.INTER_NEAREST)  # RGB
            pil_img = Image.fromarray(color_map_scaled)
            c.drawImage(ImageReader(pil_img), 0, 0, width=page_w_pt, height=page_h_pt, mask=None)

            # 轨迹（统一绿色，矢量）
            traj_col = Color(0, 1.0, 0)  # RGB 绿色（对应 BGR (0,255,0)）
            c.setStrokeColor(traj_col)
            c.setLineWidth(max(0.5, float(traj_thickness)))
            for sid, seq_w in sorted(traj_by_sid.items(), key=lambda x: x[0]):
                if len(seq_w) < 2:
                    continue
                path = c.beginPath()
                x0, y0 = self.world_xz_to_rot_px(env, seq_w[0][0], seq_w[0][1], {"map": td_map})
                X0 = x0 * scale
                Y0 = (out_h_px - y0 * scale)  # PDF 原点在左下，翻转 y
                path.moveTo(X0, Y0)
                for (xw, zw) in seq_w[1:]:
                    px, py = self.world_xz_to_rot_px(env, xw, zw, {"map": td_map})
                    X = px * scale
                    Y = (out_h_px - py * scale)
                    path.lineTo(X, Y)
                c.drawPath(path, stroke=1, fill=0)

            # 采样点（矢量圆）
            pt_col = Color(point_color[2]/255.0, point_color[1]/255.0, point_color[0]/255.0)  # BGR->RGB
            c.setStrokeColor(black)
            c.setFillColor(pt_col)
            R_pt = float(point_radius)
            for idx, (px_x, px_y) in enumerate(id_points_px):
                X = px_x * scale
                Y = (out_h_px - px_y * scale)
                c.circle(X, Y, R_pt, stroke=1, fill=1)
                if show_labels:
                    c.setFillColor(black)
                    c.setFont("Helvetica", 7.5)
                    c.drawString(X + 3, Y + 3, str(idx))
                    c.setFillColor(pt_col)

            c.showPage()
            c.save()
            print(f"[view_points] saved (PDF) -> {out_path}")
            return out_path

        # =============== 分支 B：保持原来的图片保存（同时把轨迹也画上） ===============
        vis = cv2.resize(color_map[:, :, ::-1], dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # 画点（原逻辑）
        for idx, (px_x, px_y) in enumerate(id_points_px):
            X, Y = (px_x * scale, px_y * scale)
            cv2.circle(vis, (int(X), int(Y)), point_radius, point_color, thickness=-1, lineType=cv2.LINE_AA)
            if show_labels:
                cv2.putText(vis, str(idx), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # 画轨迹（统一绿色）
        for sid, seq_w in sorted(traj_by_sid.items(), key=lambda x: x[0]):
            if len(seq_w) < 2:
                continue
            poly_px = []
            for (xw, zw) in seq_w:
                px_x, px_y = self.world_xz_to_rot_px(env, xw, zw, {"map": td_map})
                poly_px.append((int(px_x * scale), int(px_y * scale)))
            poly_px = np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly_px], isClosed=False, color=(0,255,0), thickness=int(traj_thickness), lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[view_points] saved -> {out_path}")
        return out_path





    def visualize_graph_from_json_with_keyframes_pdf(
            self,
            env,
            graph_json_path: str,
            out_pdf_path: str,
            show_nodes: bool = True,
            show_edges: bool = True,
            node_radius: int = 3,
            edge_thickness: int = 2,     # 统一边宽（pt）
            floor_y: float = 0.0,
            edge_types: Optional[List[str]] = None,  # 例: ["temporal","geometric"]
            draw_node_ids: bool = False,

            # -------- keyframe 可视化（矢量）--------
            show_keyframes: bool = False,
            draw_kf_ids: bool = False,
            draw_kf_yaw: bool = True,
            show_kf_fov: bool = True,
            kf_radius: int = 2,
            kf_arrow_len_m: float = 0.5,
            kf_fov_deg: float = 90.0,
            kf_fov_range_m: float = 0.5,

            # -------- node 半径圈（矢量填充/描边）--------
            show_node_radius: bool = False,
            radius_m: float = 0.0,
            node_radius_edge_thickness: int = 1,
            yaw_mode: str = "x_sin_z_cos",     # "x_sin_z_cos" 或 "x_cos_z_neg_sin"
            # -------- 特定位置 --------
            specific_id: Optional[int] = None,

            # -------- 其它 --------
            font_name: str = "Helvetica",
            font_size: float = 8.0,
            pt_per_px: float = 1.0,  # 页面缩放：1px≈1pt；需要更大页面可设>1
        ):
        """
        直接生成矢量 PDF（适合 Illustrator 编辑）。
        - 画布大小与 topdown 一致（不绘制 topdown）。
        - 全部以矢量路径绘制（线段、圆、折线/多边形）。
        - ReportLab 对 alpha 支持有限；如需半透明，建议在 Illustrator 中再调整。
        """
        import os, json, math
        import numpy as np
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.colors import Color, black

        # 颜色定义（BGR->近似RGB）
        col_node = Color(180/255.0, 220/255.0, 1.0)  # 节点填充
        col_edge_temporal  = Color(80/255.0, 160/255.0, 80/255.0)
        col_edge_geometric = Color(60/255.0, 140/255.0, 230/255.0)
        col_edge_default   = Color(128/255.0, 128/255.0, 180/255.0)
        col_kf_point   = Color(60/255.0, 80/255.0, 240/255.0)
        col_kf_fov     = Color(120/255.0, 160/255.0, 255/255.0)
        col_specific   = Color(150/255.0, 150/255.0, 80/255.0)
        col_node_rim   = Color(160/255.0, 200/255.0, 240/255.0)
        col_node_area  = Color(200/255.0, 220/255.0, 255/255.0)

        assert os.path.exists(graph_json_path), f"graph json not found: {graph_json_path}"
        with open(graph_json_path, "r") as f:
            G = json.load(f)

        nodes = G.get("nodes", [])
        edges = G.get("edges", [])
        meta  = G.get("meta", {}) or {}
        npz_path_in_meta = meta.get("npz_path", None)

        # 1) 仍用 topdown 取尺寸 & 坐标变换（不渲染底图）
        td_map = get_topdown_map(env.sims.pathfinder, height=float(floor_y))  # HxW uint8
        H0, W0 = td_map.shape[:2]

        # 世界->像素函数来自你的工具
        td = {"map": td_map}
        def world_to_px(xw, zw):
            x_px, y_px = self.world_xz_to_rot_px(env, xw, zw, td)
            return x_px, y_px

        # PDF 页面尺寸（pt），与像素一致后乘 pt_per_px
        page_w = W0 * pt_per_px
        page_h = H0 * pt_per_px
        os.makedirs(os.path.dirname(out_pdf_path) or ".", exist_ok=True)
        c = rl_canvas.Canvas(out_pdf_path, pagesize=(page_w, page_h))
        c.setLineCap(1)   # round
        c.setLineJoin(1)  # round

        # 将像素坐标(px)映射到 PDF 坐标(pt)。
        # 注意：PDF 原点在左下；像素原点在左上，所以需要 Y 轴翻转。
        def px_to_pt(x_px, y_px):
            return (x_px * pt_per_px, (H0 - y_px) * pt_per_px)

        # 2) 收集节点投影坐标
        id2px = {}
        id2world = {}
        for n in nodes:
            nid = int(n.get("id", len(id2px)))
            if "center" in n and isinstance(n["center"], dict):
                xw = float(n["center"].get("x", 0.0))
                zw = float(n["center"].get("z", 0.0))
            elif "pose" in n and isinstance(n["pose"], dict):
                xw = float(n["pose"].get("x", 0.0))
                zw = float(n["pose"].get("z", 0.0))
            else:
                continue
            x_px, y_px = world_to_px(xw, zw)
            id2px[nid] = (x_px, y_px)
            id2world[nid] = (xw, zw)
        if not id2px:
            raise ValueError("No nodes with valid (x,z) found.")

        # 3) 可选：node 覆盖半径（用矢量圆环/圆形）
        if show_node_radius:
            def world_meters_to_px_from(xw, zw, dx_m, dz_m):
                x2, y2 = world_to_px(xw + dx_m, zw + dz_m)
                x1, y1 = world_to_px(xw,        zw)
                return (x2 - x1, y2 - y1)

            for n in nodes:
                nid = int(n.get("id", -1))
                if (nid not in id2px) or (nid not in id2world):
                    continue
                r_m = radius_m if radius_m > 0.0 else float(n.get("radius", 0.0))
                if r_m <= 0:
                    continue
                (xw, zw) = id2world[nid]
                (cx_px, cy_px) = id2px[nid]
                dx1, dy1 = world_meters_to_px_from(xw, zw, r_m, 0.0)
                dx2, dy2 = world_meters_to_px_from(xw, zw, 0.0, r_m)
                r_px = 0.5 * (math.hypot(dx1, dy1) + math.hypot(dx2, dy2))
                # 填充圆（用淡色，透明度在 AI 中再调）
                cx_pt, cy_pt = px_to_pt(cx_px, cy_px)
                c.setFillColor(col_node_area)
                c.setStrokeColor(col_node_rim)
                c.setLineWidth(max(0.1, node_radius_edge_thickness))
                c.circle(cx_pt, cy_pt, r_px * pt_per_px, fill=1, stroke=1)

        # 4) 画边（统一边粗细）
        if show_edges:
            def color_for_type(t: str):
                t = (t or "").lower()
                if t == "temporal":  return col_edge_temporal
                if t == "geometric": return col_edge_geometric
                return col_edge_default

            c.setLineWidth(max(0.5, edge_thickness))
            for e in edges:
                u = e.get("source_id") if isinstance(e, dict) else getattr(e, "source_id", None)
                v = e.get("target_id") if isinstance(e, dict) else getattr(e, "target_id", None)
                et = e.get("type", "temporal") if isinstance(e, dict) else getattr(e, "type", "temporal")
                if u is None or v is None:
                    continue
                if (u not in id2px) or (v not in id2px):
                    continue
                if edge_types and (et not in edge_types):
                    continue

                (xu, yu) = id2px[u]; (xv, yv) = id2px[v]
                Xu, Yu = px_to_pt(xu, yu); Xv, Yv = px_to_pt(xv, yv)
                c.setStrokeColor(color_for_type(et))
                c.line(Xu, Yu, Xv, Yv)

        # 5) 节点与编号
        if show_nodes:
            for nid, (x_px, y_px) in id2px.items():
                X, Y = px_to_pt(x_px, y_px)
                c.setFillColor(col_node); c.setStrokeColor(black)
                c.circle(X, Y, node_radius * pt_per_px, fill=1, stroke=0)
                if draw_node_ids:
                    c.setFillColor(black)
                    c.setFont(font_name, font_size)
                    c.drawString(X + 4 * pt_per_px, Y + 2 * pt_per_px, str(nid))

        # 6) Keyframes：点、箭头、FOV（矢量）
        fid2_poseyaw = {}
        if show_keyframes:
            if npz_path_in_meta and os.path.exists(npz_path_in_meta):
                data = np.load(npz_path_in_meta, allow_pickle=True)
                frames_all = data["frame_ids"].astype(np.int64)
                xyz_all    = data["poses_xyz"].astype(np.float32)   # [N,3]
                yaws_all   = data["yaws"].astype(np.float32)        # [N]
                for i, fid in enumerate(frames_all):
                    fid2_poseyaw[int(fid)] = (float(xyz_all[i,0]), float(xyz_all[i,2]), float(yaws_all[i]))
            else:
                print("[viz-graph-json] WARNING: meta.npz_path 不存在，无法绘制 keyframe。")

            def draw_one_kf(xw, zw, yaw_rad, kfid_text=None):
                px, py = world_to_px(xw, zw)
                X, Y = px_to_pt(px, py)
                # 点
                c.setFillColor(col_kf_point); c.circle(X, Y, kf_radius*pt_per_px, fill=1, stroke=0)
                if draw_kf_ids and kfid_text is not None:
                    c.setFillColor(black); c.setFont(font_name, font_size)
                    c.drawString(X + 3*pt_per_px, Y + 2*pt_per_px, str(kfid_text))

                # 箭头
                if draw_kf_yaw:
                    if yaw_mode == "x_sin_z_cos":
                        dx_m = math.sin(yaw_rad) * kf_arrow_len_m
                        dz_m = math.cos(yaw_rad) * kf_arrow_len_m
                    else:
                        dx_m = math.cos(yaw_rad) * kf_arrow_len_m
                        dz_m = -math.sin(yaw_rad) * kf_arrow_len_m
                    x2, y2 = world_to_px(xw + dx_m, zw + dz_m)
                    X2, Y2 = px_to_pt(x2, y2)
                    # 画箭头：主线 + 简单箭头三角
                    c.setStrokeColor(col_kf_point); c.setLineWidth(1)
                    c.line(X, Y, X2, Y2)
                    # 箭头端小三角
                    ang = math.atan2(Y2 - Y, X2 - X)
                    ah = 5 * pt_per_px
                    left  = (X2 - ah*math.cos(ang - 0.35), Y2 - ah*math.sin(ang - 0.35))
                    right = (X2 - ah*math.cos(ang + 0.35), Y2 - ah*math.sin(ang + 0.35))
                    p = c.beginPath(); p.moveTo(X2, Y2); p.lineTo(*left); p.lineTo(*right); p.close()
                    c.setFillColor(col_kf_point); c.drawPath(p, stroke=0, fill=1)

                # FOV 扇形（近似三角扇）
                if show_kf_fov:
                    half = math.radians(kf_fov_deg * 0.5)
                    if yaw_mode == "x_sin_z_cos":
                        dx_l = math.sin(yaw_rad + half) * kf_fov_range_m
                        dz_l = math.cos(yaw_rad + half) * kf_fov_range_m
                        dx_r = math.sin(yaw_rad - half) * kf_fov_range_m
                        dz_r = math.cos(yaw_rad - half) * kf_fov_range_m
                    else:
                        dx_l = math.cos(yaw_rad + half) * kf_fov_range_m
                        dz_l = -math.sin(yaw_rad + half) * kf_fov_range_m
                        dx_r = math.cos(yaw_rad - half) * kf_fov_range_m
                        dz_r = -math.sin(yaw_rad - half) * kf_fov_range_m
                    xl, yl = world_to_px(xw + dx_l, zw + dz_l)
                    xr, yr = world_to_px(xw + dx_r, zw + dz_r)
                    Xl, Yl = px_to_pt(xl, yl); Xr, Yr = px_to_pt(xr, yr)
                    p = c.beginPath()
                    p.moveTo(X, Y); p.lineTo(Xl, Yl); p.lineTo(Xr, Yr); p.close()
                    c.setFillColor(col_kf_fov)   # 透明度后续在 AI 里调
                    c.drawPath(p, stroke=0, fill=1)

            # 遍历
            for n in nodes:
                kfs = n.get("keyframes", [])
                if not kfs:
                    continue
                for kfid in kfs:
                    rec = fid2_poseyaw.get(int(kfid))
                    if rec is None:
                        continue
                    xw, zw, yaw = rec
                    draw_one_kf(xw, zw, yaw, kfid_text=kfid)

        # 7) 特定帧点
        if specific_id is not None:
            if specific_id not in (fid2_poseyaw or {}):
                if npz_path_in_meta and os.path.exists(npz_path_in_meta):
                    data = np.load(npz_path_in_meta, allow_pickle=True)
                    frames_all = data["frame_ids"].astype(np.int64)
                    xyz_all    = data["poses_xyz"].astype(np.float32)
                    yaws_all   = data["yaws"].astype(np.float32)
                    for i, fid in enumerate(frames_all):
                        fid2_poseyaw[int(fid)] = (float(xyz_all[i,0]), float(xyz_all[i,2]), float(yaws_all[i]))
            rec = (fid2_poseyaw or {}).get(int(specific_id))
            if rec is not None:
                xw, zw, yaw = rec
                px, py = world_to_px(xw, zw)
                X, Y = px_to_pt(px, py)
                c.setFillColor(col_specific); c.circle(X, Y, kf_radius*pt_per_px, fill=1, stroke=0)

        # 8) 收尾
        c.showPage()
        c.save()
        print(f"[viz-graph-json] saved vector PDF -> {out_pdf_path}")

    def visualize_graph_from_json_with_keyframes_real_robot(
            self,
            env,  # 保留但不使用，兼容旧接口
            graph_json_path: str,
            out_path: str,
            show_nodes: bool = True,
            show_edges: bool = True,
            node_radius: int = 3,
            edge_thickness: int = 1,
            canvas_h: int = 1024,
            floor_y: float = 0.0,  # 不用
            edge_types=None,
            draw_node_ids: bool = False,

            # -------- keyframe 可视化 --------
            show_keyframes: bool = False,
            draw_kf_ids: bool = False,
            draw_kf_yaw: bool = True,
            show_kf_fov: bool = True,
            kf_radius: int = 2,
            kf_arrow_len_m: float = 0.5,
            kf_fov_deg: float = 90.0,
            kf_fov_range_m: float = 0.5,
            kf_fov_alpha: float = 0.28,
            kf_color: tuple = (60, 80, 240),        # BGR
            kf_fov_color: tuple = (120, 160, 255),  # BGR
            yaw_mode: str = "x_sin_z_cos",

            # -------- node 半径圈 --------
            show_node_radius: bool = False,
            node_radius_alpha: float = 0.15,
            node_radius_fill_color: tuple = (200, 220, 255),
            node_radius_edge_color: tuple = (160, 200, 240),
            node_radius_edge_thickness: int = 1,
            radius_m: float = 0.0,

            # -------- 特定位置可视化 -------
            specific_id: int = 0,

            # -------- 新增：画布参数 --------
            canvas_w: int = None,           # 默认与 canvas_h 等比例
            margin_px: int = 40,
            bg_color: tuple = (255, 255, 255),  # 白底 BGR
    ):
        import os, json, math
        import numpy as np
        import cv2

        assert os.path.exists(graph_json_path), f"graph json not found: {graph_json_path}"
        with open(graph_json_path, "r") as f:
            G = json.load(f)

        nodes = G.get("nodes", [])
        edges = G.get("edges", [])
        meta  = G.get("meta", {}) or {}
        npz_path_in_meta = meta.get("npz_path", None)

        # -------------------------
        # 1) 读取 keyframe 用的 pose/yaw（从 npz）
        # -------------------------
        fid2_poseyaw = {}
        if show_keyframes or (specific_id is not None):
            if npz_path_in_meta and os.path.exists(npz_path_in_meta):
                data = np.load(npz_path_in_meta, allow_pickle=True)
                frames_all = data["frame_ids"].astype(np.int64)
                xyz_all    = data["poses_xyz"].astype(np.float32)   # [N,3]
                yaws_all   = data["yaws"].astype(np.float32)        # [N]
                for i, fid in enumerate(frames_all):
                    # 平面用 (x,z)
                    fid2_poseyaw[int(fid)] = (float(xyz_all[i,0]), float(xyz_all[i,2]), float(yaws_all[i]))
            else:
                print("[viz] WARNING: meta.npz_path 不存在，keyframe/specific 无法从 npz 取位姿。")

        # -------------------------
        # 2) 收集需要绘制的世界坐标点，用于确定画布范围
        #    平面 = (x, z)
        # -------------------------
        pts_xz = []

        # nodes 的中心
        id2world = {}  # node_id -> (x,z)
        for n in nodes:
            nid = int(n.get("id", -1))
            if nid < 0:
                continue
            if "center" in n and isinstance(n["center"], dict):
                xw = float(n["center"].get("x", 0.0))
                zw = float(n["center"].get("z", 0.0))
            elif "pose" in n and isinstance(n["pose"], dict):
                xw = float(n["pose"].get("x", 0.0))
                zw = float(n["pose"].get("z", 0.0))
            else:
                continue
            id2world[nid] = (xw, zw)
            pts_xz.append((xw, zw))

        # keyframes 的点（如果画 keyframes，需要把它们也计入范围）
        if show_keyframes:
            for n in nodes:
                for kfid in n.get("keyframes", []) or []:
                    rec = fid2_poseyaw.get(int(kfid))
                    if rec is None:
                        continue
                    xw, zw, _ = rec
                    pts_xz.append((xw, zw))

        # specific 点也计入范围
        if specific_id is not None:
            rec = fid2_poseyaw.get(int(specific_id))
            if rec is not None:
                xw, zw, _ = rec
                pts_xz.append((xw, zw))

        if len(pts_xz) == 0:
            raise RuntimeError("[viz] No points found to visualize (nodes/keyframes empty or missing).")

        xs = np.array([p[0] for p in pts_xz], dtype=np.float32)
        zs = np.array([p[1] for p in pts_xz], dtype=np.float32)

        x_min, x_max = float(xs.min()), float(xs.max())
        z_min, z_max = float(zs.min()), float(zs.max())

        # 防止范围为0导致除零
        eps = 1e-6
        if abs(x_max - x_min) < eps:
            x_max += 1.0
            x_min -= 1.0
        if abs(z_max - z_min) < eps:
            z_max += 1.0
            z_min -= 1.0

        # -------------------------
        # 3) 创建画布 + 坐标映射 world(x,z) -> pixel(X,Y)
        #    让 x 向右，z 向上（像素 y 是向下，所以要翻一下）
        # -------------------------
        if canvas_w is None:
            canvas_w = canvas_h  # 方形画布更直观

        vis = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

        # 留边距
        W = canvas_w - 2 * margin_px
        H = canvas_h - 2 * margin_px

        def world_to_px(xw, zw):
            # x -> [margin, margin+W]
            u = (xw - x_min) / (x_max - x_min)
            v = (zw - z_min) / (z_max - z_min)
            X = margin_px + u * W
            # z 越大越“上”，像素 y 越小
            Y = margin_px + (1.0 - v) * H
            return float(X), float(Y)

        # 米到像素的近似缩放（用于箭头/扇形/半径）
        # 用 x/z 两个方向的比例取平均，保证稳定
        px_per_m_x = W / (x_max - x_min)
        px_per_m_z = H / (z_max - z_min)
        px_per_m = 0.5 * (px_per_m_x + px_per_m_z)

        # -------------------------
        # 4) 画 node 覆盖半径圈（可选）
        # -------------------------
        if show_node_radius:
            overlay = vis.copy()
            for n in nodes:
                nid = int(n.get("id", -1))
                if nid not in id2world:
                    continue
                xw, zw = id2world[nid]
                if radius_m == 0.0:
                    r_m = float(n.get("radius", 0.0))
                else:
                    r_m = float(radius_m)
                if r_m <= 0:
                    continue

                X, Y = world_to_px(xw, zw)
                R = max(1, int(round(r_m * px_per_m)))

                cv2.circle(overlay, (int(X), int(Y)), R, node_radius_fill_color,
                        thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(vis, (int(X), int(Y)), R, node_radius_edge_color,
                        thickness=node_radius_edge_thickness, lineType=cv2.LINE_AA)

            cv2.addWeighted(overlay, node_radius_alpha, vis, 1 - node_radius_alpha, 0, dst=vis)

        # -------------------------
        # 5) 画边
        # -------------------------
        if show_edges:
            def color_for_type(t: str):
                t = (t or "").lower()
                if t == "temporal":  return (80, 160, 80)
                if t == "geometric": return (60, 140, 230)
                return (128, 128, 180)

            for e in edges:
                u = e.get("source_id") if isinstance(e, dict) else getattr(e, "source_id", None)
                v = e.get("target_id") if isinstance(e, dict) else getattr(e, "target_id", None)
                et = e.get("type", "temporal") if isinstance(e, dict) else getattr(e, "type", "temporal")

                cnt = e.get("count", None) if isinstance(e, dict) else getattr(e, "count", None)
                if cnt is None:
                    dp = e.get("delta_pose", {}) if isinstance(e, dict) else getattr(e, "delta_pose", {}) or {}
                    cnt = dp.get("count", 1)

                if u is None or v is None:
                    continue
                u = int(u); v = int(v)
                if (u not in id2world) or (v not in id2world):
                    continue
                if edge_types and (et not in edge_types):
                    continue

                (xu, zu) = id2world[u]
                (xv, zv) = id2world[v]
                Xu, Yu = world_to_px(xu, zu)
                Xv, Yv = world_to_px(xv, zv)

                t = int(max(1, edge_thickness + min(4, int(math.log1p(max(1, int(cnt)))))))
                cv2.line(vis, (int(Xu), int(Yu)), (int(Xv), int(Yv)),
                        color_for_type(et), thickness=t, lineType=cv2.LINE_AA)

        # -------------------------
        # 6) 画节点
        # -------------------------
        if show_nodes:
            for nid, (xw, zw) in id2world.items():
                X, Y = world_to_px(xw, zw)
                cv2.circle(vis, (int(X), int(Y)), node_radius, (180, 220, 255),
                        thickness=-1, lineType=cv2.LINE_AA)
                if draw_node_ids:
                    cv2.putText(vis, str(nid), (int(X)+4, int(Y)-4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------
        # 7) 画 keyframes（点 + yaw箭头 + FOV扇形）
        # -------------------------
        def draw_one_kf(xw, zw, yaw_rad, kfid_text=None):
            X, Y = world_to_px(xw, zw)
            cv2.circle(vis, (int(X), int(Y)), kf_radius, kf_color, thickness=-1, lineType=cv2.LINE_AA)
            if draw_kf_ids and kfid_text is not None:
                cv2.putText(vis, str(kfid_text), (int(X)+3, int(Y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, kf_color, 1, cv2.LINE_AA)

            # yaw 箭头
            if draw_kf_yaw:
                if yaw_mode == "x_sin_z_cos":
                    dx = math.sin(yaw_rad) * kf_arrow_len_m
                    dz = math.cos(yaw_rad) * kf_arrow_len_m
                else:  # "x_cos_z_neg_sin"
                    dx = math.cos(yaw_rad) * kf_arrow_len_m
                    dz = -math.sin(yaw_rad) * kf_arrow_len_m

                X2, Y2 = world_to_px(xw + dx, zw + dz)
                cv2.arrowedLine(vis, (int(X), int(Y)), (int(X2), int(Y2)), kf_color, 1, tipLength=0.18)

            # FOV 扇形（三角形近似）
            if show_kf_fov:
                half = math.radians(kf_fov_deg * 0.5)
                if yaw_mode == "x_sin_z_cos":
                    dx_l = math.sin(yaw_rad + half) * kf_fov_range_m
                    dz_l = math.cos(yaw_rad + half) * kf_fov_range_m
                    dx_r = math.sin(yaw_rad - half) * kf_fov_range_m
                    dz_r = math.cos(yaw_rad - half) * kf_fov_range_m
                else:
                    dx_l = math.cos(yaw_rad + half) * kf_fov_range_m
                    dz_l = -math.sin(yaw_rad + half) * kf_fov_range_m
                    dx_r = math.cos(yaw_rad - half) * kf_fov_range_m
                    dz_r = -math.sin(yaw_rad - half) * kf_fov_range_m

                Xl, Yl = world_to_px(xw + dx_l, zw + dz_l)
                Xr, Yr = world_to_px(xw + dx_r, zw + dz_r)

                overlay = vis.copy()
                pts = np.array([[int(X), int(Y)], [int(Xl), int(Yl)], [int(Xr), int(Yr)]], dtype=np.int32)
                cv2.fillConvexPoly(overlay, pts, kf_fov_color)
                cv2.addWeighted(overlay, kf_fov_alpha, vis, 1 - kf_fov_alpha, 0, dst=vis)

        if show_keyframes:
            for n in nodes:
                kfs = n.get("keyframes", [])
                if not kfs:
                    continue
                for kfid in kfs:
                    rec = fid2_poseyaw.get(int(kfid))
                    if rec is None:
                        continue
                    xw, zw, yaw = rec
                    draw_one_kf(xw, zw, yaw, kfid_text=kfid)

        # -------------------------
        # 8) 高亮 specific_id
        # -------------------------
        if specific_id is not None:
            rec = fid2_poseyaw.get(int(specific_id))
            if rec is not None:
                xw, zw, yaw = rec
                X, Y = world_to_px(xw, zw)
                specific_color = (150, 150, 80)  # BGR
                cv2.circle(vis, (int(X), int(Y)), max(2, kf_radius+1), specific_color,
                        thickness=-1, lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[viz-graph-json-no-topdown] saved -> {out_path}")
        return out_path

    def visualize_view_points_real_robot(
            self,
            env,  # 保留但不使用，兼容旧接口
            frames_meta_jsonl_path: str,
            out_path: str,
            filter_y_range=(None, None),
            point_color=(0, 0, 255),     # BGR
            point_radius: int = 5,
            show_labels: bool = False,
            canvas_h: int = 1024,
            y_floor: float = 0.0,        # 不用
            canvas_w: int = None,
            margin_px: int = 40,
            bg_color=(255, 255, 255),    # 白底
            axis_color=(80, 80, 80),     # 坐标轴颜色
            draw_axes: bool = True,
    ):
        """
        在纯画布上可视化采集的 view 点位置（不依赖 Habitat topdown map）。
        - 从 frames_meta.jsonl 读取 pose.x/y/z
        - 可按 y 过滤
        - 在 x-z 平面画点（x 向右，z 向上）
        """
        import os, json
        import numpy as np
        import cv2

        assert os.path.exists(frames_meta_jsonl_path), f"jsonl文件不存在: {frames_meta_jsonl_path}"

        # 1) 读取点（默认用 x-z 平面）
        points = []  # [(x,z)]
        with open(frames_meta_jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                pose = data.get("pose", {}) or {}
                x = float(pose.get("x", 0.0))
                y = float(pose.get("y", 0.0))
                z = float(pose.get("z", 0.0))

                lo, hi = filter_y_range
                if (lo is not None and y < lo) or (hi is not None and y > hi):
                    continue

                points.append((x, z))

        if len(points) == 0:
            raise RuntimeError("[view_points] No points after filtering.")

        xs = np.array([p[0] for p in points], dtype=np.float32)
        zs = np.array([p[1] for p in points], dtype=np.float32)

        x_min, x_max = float(xs.min()), float(xs.max())
        z_min, z_max = float(zs.min()), float(zs.max())

        # 防止范围过小导致除零
        eps = 1e-6
        if abs(x_max - x_min) < eps:
            x_min -= 1.0
            x_max += 1.0
        if abs(z_max - z_min) < eps:
            z_min -= 1.0
            z_max += 1.0

        # 2) 创建画布
        if canvas_w is None:
            canvas_w = canvas_h
        vis = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

        W = canvas_w - 2 * margin_px
        H = canvas_h - 2 * margin_px

        # 3) world(x,z) -> pixel(X,Y)
        # x 向右；z 向上（像素 y 向下，所以要翻）
        def world_to_px(xw, zw):
            u = (xw - x_min) / (x_max - x_min)
            v = (zw - z_min) / (z_max - z_min)
            X = margin_px + u * W
            Y = margin_px + (1.0 - v) * H
            return float(X), float(Y)

        # 4) 可选画坐标轴（x=0 和 z=0 的位置）
        if draw_axes:
            # x=0 竖线（如果 0 在范围内）
            if x_min <= 0.0 <= x_max:
                X0, _ = world_to_px(0.0, z_min)
                X1, _ = world_to_px(0.0, z_max)
                cv2.line(vis, (int(X0), margin_px), (int(X1), canvas_h - margin_px), axis_color, 1, cv2.LINE_AA)
                cv2.putText(vis, "x=0", (int(X0)+4, margin_px+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1, cv2.LINE_AA)

            # z=0 横线（如果 0 在范围内）
            if z_min <= 0.0 <= z_max:
                _, Y0 = world_to_px(x_min, 0.0)
                _, Y1 = world_to_px(x_max, 0.0)
                cv2.line(vis, (margin_px, int(Y0)), (canvas_w - margin_px, int(Y1)), axis_color, 1, cv2.LINE_AA)
                cv2.putText(vis, "z=0", (margin_px+4, int(Y0)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1, cv2.LINE_AA)

        # 5) 画点
        for idx, (xw, zw) in enumerate(points):
            X, Y = world_to_px(xw, zw)
            cv2.circle(vis, (int(X), int(Y)), point_radius, point_color, thickness=-1, lineType=cv2.LINE_AA)
            if show_labels:
                cv2.putText(vis, str(idx), (int(X)+4, int(Y)-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        # 6) 右下角写范围信息（方便 debug）
        info = f"x[{x_min:.2f},{x_max:.2f}]  z[{z_min:.2f},{z_max:.2f}]  N={len(points)}"
        cv2.putText(vis, info, (margin_px, canvas_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 7) 保存
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[view_points-no-topdown] saved -> {out_path}")
        return out_path


    #################
    ###### FrontierExplorer
    #################

    def _show_map(self, obs):
        # 获取 RGB 图像并转换为 BGR 格式
        bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)

        # 获取深度图像，并归一化到 0-255 范围以便显示
        depth = obs["depth"]
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # cv map
        cv_map = cv2.cvtColor(self.cv_map, cv2.COLOR_RGB2BGR)
        FrontierMap = cv2.cvtColor(self.FrontierMap, cv2.COLOR_RGB2BGR)

        # 将三种图像调整为相同大小
        h, w = bgr.shape[:2]
        depth_resized = cv2.resize(depth_normalized, (w, h))
        cv_map_resized = cv2.resize(cv_map, (w, h))
        FrontierMap_resized = cv2.resize(FrontierMap, (w, h))
        
        # 合并 RGB、深度和语义图像
        combined_image = np.hstack((bgr, cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR), cv_map_resized, FrontierMap_resized))

        # 调整窗口大小，使其小于屏幕宽度
        screen_width = 6200  # 假设屏幕宽度为 800 像素
        scale_factor = min(screen_width / combined_image.shape[1], 1.0)
        new_width = int(combined_image.shape[1] * scale_factor)
        new_height = int(combined_image.shape[0] * scale_factor)
        combined_image_resized = cv2.resize(combined_image, (new_width, new_height))
        
        # 显示拼接图像
        cv2.imshow("RGB | Depth | Semantic | cvmap | FrontierMap | detect", combined_image_resized)

    def save_final_maps(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        cv_map_bgr = cv2.cvtColor(self.cv_map, cv2.COLOR_RGB2BGR)
        frontier_bgr = cv2.cvtColor(self.FrontierMap, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"cv_map.png"), cv_map_bgr)
        cv2.imwrite(os.path.join(out_dir, f"frontier_map.png"), frontier_bgr)
        side = np.hstack([cv_map_bgr, cv2.resize(frontier_bgr, (cv_map_bgr.shape[1], cv_map_bgr.shape[0]))])
        cv2.imwrite(os.path.join(out_dir, f"cv_and_frontier.png"), side)

    def grid2loc_2d(self, x, y):
        row, col = x, y
        initial_position = self.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.gs // 2) * self.cs
        actual_x = initial_x + (col - self.gs // 2) * self.cs
        actual_z = initial_z
        
        # 返回实际坐标
        return np.array([actual_x, actual_z, actual_y])
    
    def loc2grid_2d(self, x_base, y_base):
        row = int(self.gs / 2 - int(x_base / self.cs))
        col = int(self.gs / 2 - int(y_base / self.cs))
        return row, col
    
    def is_unknown(self, x: int, y: int) -> bool:
        return (self.cv_map[x, y].sum() == 0)   
    def is_known(self, x: int, y: int) -> bool:
        return not self.is_unknown(x, y)     
    def in_bounds(self, x: int, y: int) -> bool:
        return (0 <= x < self.gs and 0 <= y < self.gs)
    def is_navigabale(self, x: int, y: int) -> bool:
        return self.Env.plnner.pathfinder.is_navigable(self.grid2loc_2d(x,y))    
        
    def build_navigable_mask(self) -> np.ndarray:
        """
        根据 self.map_3d 构建一个 [gs, gs] 的布尔数组，表示哪些网格可导航。
        """
        navigable_mask = np.zeros((self.gs, self.gs), dtype=bool)
        for x in range(self.gs):
            for y in range(self.gs):
                # 如果某格是已知，则认为可导航
                if self.is_known(x, y) and self.is_navigabale(x, y):
                    navigable_mask[x, y] = True
        return navigable_mask
    
    def find_frontiers(self, navigable_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        寻找当前地图中的所有前沿点(网格坐标)。
        前沿点定义：已知+可导航，且与至少一个未知邻居相邻。
        :return: list of (x, y)，表示前沿点的网格坐标
        """
        frontiers = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x in range(self.gs):
            for y in range(self.gs):
                if not navigable_mask[x, y]:
                    continue
                if self.is_known(x, y):
                    # 判断是否与未知邻居相邻
                    neighbors_unknown = False
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if self.in_bounds(nx, ny) and self.is_unknown(nx, ny):
                            neighbors_unknown = True
                            break
                    if neighbors_unknown:
                        frontiers.append((x, y))
        return frontiers    
    
    def cluster_frontiers(self, frontiers: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        对前沿点进行连通性聚类，返回若干个簇，每个簇是一组前沿点 (x, y)。
        使用4邻域BFS。
        """
        if not frontiers:
            return []

        frontier_set = set(frontiers)
        visited = set()
        clusters = []

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for f in frontiers:
            if f in visited:
                continue
            # BFS
            queue = deque([f])
            cluster = []
            visited.add(f)

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            clusters.append(cluster)
            
        filtered_clusters = []
        for c in clusters:
            if len(c) >= self.min_cluster_size:
                filtered_clusters.append(c)

        return filtered_clusters

    
    def compute_cluster_center(self, cluster: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        计算前沿簇的中心（例如质心），并返回 (cx, cy) 浮点数网格坐标。
        """
        cx = sum([p[0] for p in cluster]) / len(cluster)
        cy = sum([p[1] for p in cluster]) / len(cluster)
        return (cx, cy)
    
    def compute_information_gain(self, center_x: float, center_y: float) -> float:
        """
        在网格坐标系下，以 (center_x, center_y) 为中心，查看半径 ig_radius 范围内未知格子数量，
        作为信息增益近似值。
        """
        cx = int(round(center_x))
        cy = int(round(center_y))

        unknown_count = 0
        radius = self.ig_radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = cx + dx
                ny = cy + dy
                if not self.in_bounds(nx, ny):
                    continue
                if self.is_unknown(nx, ny):
                    unknown_count += 1

        return float(unknown_count)
    
    def select_best_cluster_center_by_ig(
        self,
        frontier_clusters: List[List[Tuple[int, int]]]
    ) -> Tuple[float, float]:
        """
        基于信息增益来选择最优的前沿簇中心。
        具体做法：
          1. 对每个簇，先计算其网格质心 (cx, cy)
          2. 估算信息增益: compute_information_gain(cx, cy)
          3. 选出信息增益最高的簇中心。
        
        如果所有簇都0信息增益,可以返回None表示没有值得去的前沿点。
        """
        best_center = None
        best_ig = 0.0

        for cluster in frontier_clusters:
            cx, cy = self.compute_cluster_center(cluster)
            ig = self.compute_information_gain(cx, cy)
            if ig > best_ig:
                best_ig = ig
                best_center = (cx, cy)

        # 如果信息增益全是 0，则返回 None 代表没有价值的探索目标
        if best_center is None or best_ig == 0:
            return None

        return best_center

    def update_frontier_map(self, frontiers, frontier_clusters, target_center_map):
        frontier_img = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)

        # 1) 已知/未知/可导航 上色
        for x in range(self.gs):
            for y in range(self.gs):
                if self.is_unknown(x, y):
                    frontier_img[x, y] = (0, 0, 0)           # 黑:未知
                else:
                    if self.is_navigabale(x, y):
                        frontier_img[x, y] = (255, 255, 255) # 白:已知可导航
                    else:
                        frontier_img[x, y] = (100, 100, 100) # 灰:已知不可导航
        
        # 2) 标记前沿点 (红色)
        for (fx, fy) in frontiers:
            frontier_img[fx, fy] = (255, 0, 0)

        # # 3) 随机颜色标记前沿簇
        # for cluster in frontier_clusters:
        #     color = (
        #         random.randint(50, 255),
        #         random.randint(50, 255),
        #         random.randint(50, 255)
        #     )
        #     for (cx, cy) in cluster:
        #         frontier_img[cx, cy] = color

        # 4) 标记目标点(绿色)
        if target_center_map is not None:
            tx = int(round(target_center_map[0]))
            ty = int(round(target_center_map[1]))
            if 0 <= tx < self.gs and 0 <= ty < self.gs:
                cv2.circle(frontier_img, (ty, tx), 5, (0, 255, 0), -1)

        self.FrontierMap = frontier_img

    def _out_of_range(self, row, col, height):        
        return col >= self.gs or row >= self.gs or height >= self.maxh or col < 0 or row < 0 or height < self.minh

    def _backproject_depth(self, depth):
        
        pc, mask = depth2pc(depth, intr_mat=self.calib_mat, min_depth=self.min_depth, max_depth=self.max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::self.depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc
    
    def cvmap_update(self, obs, pose):
        """
        简化版：只更新 cv_map / max_height（不提取 patch token，不维护 grid_feat 等）
        目的：让 is_unknown() 通过 cv_map 是否为 0 来判断“是否被观测过”
        """
        # (1) 初始化初始位姿
        if len(self.inv_init_base_tf) == 0:
            self.init_base_tf = cvt_pose_vec2tf(pose)
            self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        
        # （2） 当前位姿到“以初始为原点”的变换
        habitat_base_pose = cvt_pose_vec2tf(pose)
        base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
        self.tf = self.inv_init_base_tf @ base_pose
        
        # （3）提取传感器数据
        rgb = np.array(obs["rgb"][:,:,:3])
        depth = np.array(obs["depth"])
        
        # （4）反投影：相机点云 ————> 世界系坐标
        pc_cam = self._backproject_depth(depth)
        pc_transform = self.tf @ self.base_transform @ self.base2cam_tf
        pc_global = transform_pc(pc_cam, pc_transform) 
        
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc_cam.T)):
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
            if self._out_of_range(row, col, height):
                continue
            height = height - self.minh
            
            px, py, pz = project_point(self.calib_mat, p_local)
            rgb_v = rgb[py, px, :]

            if height >= self.max_height[row, col]:
                self.max_height[row, col] = height
                self.cv_map[row, col] = rgb_v
    
    def explore_frontier(self, max_iterations=30, turn_left_deg: float = 30.0,
                        save_rgb: Literal["none", "stride", "all"] = "none",
                        rgb_stride: int = 30, rgb_format: Literal["jpg", "png"] = "jpg",
                        save_depth: Literal["none", "stride", "all"] = "none",
                        depth_stride: int = 30, depth_format: Literal["png16", "npy"] = "png16",
                        depth_scale: float = 1000.0):
        """
        前沿点探索的主方法：
        不断寻找前沿点 -> 聚类 -> 选目标 -> 导航，直到前沿点耗尽或达到最大迭代次数。
        """
        # 1) 准备 env
        env = self.Env

        # 2) 创建 saver（与 explore() 一致）
        self.obs_saver = ObservationSaver(
            root_dir=self.save_dir,
            save_rgb=save_rgb, rgb_stride=rgb_stride, rgb_format=rgb_format,
            save_depth=save_depth, depth_stride=depth_stride,
            depth_format=depth_format, depth_scale=depth_scale
        )
        # self.sem_saver = SemanticSaver(self.save_dir)

        # 3) 初始化参数
        self.min_cluster_size = 10
        self.ig_radius = 5

        # 4) episode id
        if not hasattr(self, "_episode_id"):
            self._episode_id = 0
        self._episode_id += 1
        episode_id = self._episode_id

        # 5) 初帧
        obs = env.sims.get_sensor_observations(0)
        agent_state = env.agent.get_state()
        if "rgb" not in obs:
            raise RuntimeError("Observation does not contain 'rgb'.")
        rgb = obs["rgb"]
        depth = obs.get("depth", None)
        self._record_step(rgb, depth, agent_state, meta={
            "episode_id": int(episode_id),
            "subgoal_id": int(-1),
            "traj_step": int(0),
            "action": "init"
        })

        # 6) 主循环：frontier -> cluster -> 选目标 -> 导航（每一步都按 explore() 的方式存储）
        iteration_count = 0
        traj_step = 1

        while iteration_count < max_iterations:
            print("iteration_count:", iteration_count)
            iteration_count += 1

            # 6.1 原地转圈扫描
            n_turns = int(round(360.0 / float(turn_left_deg)))
            for _t in range(max(1, n_turns)):
                obs = env.sims.step("turn_left")
                agent_state = env.agent.get_state()
                rgb = obs["rgb"]; depth = obs.get("depth", None)
                self._record_step(rgb, depth, agent_state, meta={
                    "episode_id": int(episode_id),
                    "subgoal_id": int(iteration_count - 1),
                    "traj_step": int(traj_step),
                    "action": "turn_left",
                })
                traj_step += 1

                # **只更新 cv_map**（简化后的 obs2voxeltoken）
                pos, rot = agent_state.position, agent_state.rotation
                pose = np.array([pos[0], pos[1], pos[2], rot.x, rot.y, rot.z, rot.w], dtype=np.float32)
                self.cvmap_update(obs, pose)

            if iteration_count % 1 == 0:
                print("known pixels:", int((self.cv_map.sum(axis=2)>0).sum()))


            # 6.2 构建可导航/前沿
            navigable_mask = self.build_navigable_mask()
            frontiers = self.find_frontiers(navigable_mask)
            if not frontiers:
                print("No Frontiers, Stop!")
                break
            frontier_clusters = self.cluster_frontiers(frontiers)
            if not frontier_clusters:
                print("No Frontier clusters, Stop!")
                break

            # 6.3 选目标
            target_center_map = self.select_best_cluster_center_by_ig(frontier_clusters)
            if target_center_map is None:
                break

            self.update_frontier_map(frontiers, frontier_clusters, target_center_map)

            # 6.4 导航到 subgoal（保持你的 grid2loc_2d + random_navigable_near + move2point）
            subgoal = self.grid2loc_2d(target_center_map[0], target_center_map[1])
            subgoal = self.Env.plnner.pathfinder.get_random_navigable_point_near(subgoal, radius= 1.50)
            try:
                path, _goal = env.move2point(subgoal)
            except Exception as e:
                print("[WARN] move2point failed:", e)
                continue

            # 6.5 执行动作并记录
            for act in path:
                if act == "stop":
                    continue
                obs = env.sims.step(act)
                agent_state = env.agent.get_state()
                rgb = obs["rgb"]; depth = obs.get("depth", None)

                self._record_step(rgb, depth, agent_state, meta={
                    "episode_id": int(episode_id),
                    "subgoal_id": int(iteration_count - 1),
                    "traj_step": int(traj_step),
                    "action": str(act),
                })
                traj_step += 1

                # 同步更新 cv_map（简化函数）
                pos, rot = agent_state.position, agent_state.rotation
                pose = np.array([pos[0], pos[1], pos[2], rot.x, rot.y, rot.z, rot.w], dtype=np.float32)
                self.cvmap_update(obs, pose)

            # 记录当前高度
            self.base_height.append(env.agent.get_state().position[1])

        # 7) 结束与落盘（与 explore() 一致）
        self._save_npz("explore_log.npz")
        if self.obs_saver is not None: self.obs_saver.close()
        # if self.sem_saver is not None: self.sem_saver.close()
        np.save(self.save_dir + "/base_height.npy", np.array(self.base_height))
        map_address = self.args.memory_path +  '/' + self.args.scene_name
        self.save_final_maps(map_address)
 
# -----------------------------
# CLI
# -----------------------------
def _default_args_like():
    """Create a minimal args-like object with default fields used by this builder."""
    class A: pass
    a = A()
    a.dino_size = "dinov2_vitb14"
    a.memory_path = "./memory_out"
    a.scene_name = "scene"
    return a

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scene_name", type=str, default="scene")
    p.add_argument("--memory_path", type=str, default="./memory_out")
    p.add_argument("--dino_size", type=str, default="dinov2_vitb14")

    # exploration
    p.add_argument("--random_move_num", type=int, default=50)
    p.add_argument("--turn_left_deg", type=float, default=30.0)
    p.add_argument("--lock_floor", action="store_true")
    p.add_argument("--floor_band_m", type=float, default=0.30)

    # observation saving policy
    p.add_argument("--save_rgb", type=str, choices=["none", "stride", "all"], default="none")
    p.add_argument("--rgb_stride", type=int, default=30)
    p.add_argument("--rgb_format", type=str, choices=["jpg", "png"], default="jpg")
    p.add_argument("--save_depth", type=str, choices=["none", "stride", "all"], default="none")
    p.add_argument("--depth_stride", type=int, default=30)
    p.add_argument("--depth_format", type=str, choices=["png16", "npy"], default="png16")
    p.add_argument("--depth_scale", type=float, default=1000.0)

    # graph
    p.add_argument("--cos_thresh", type=float, default=0.18)
    p.add_argument("--yaw_gate_deg", type=float, default=30.0)
    p.add_argument("--tau_len", type=int, default=2)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--r_max", type=float, default=6.0)

    args = p.parse_args()

    class ARGS:
        pass
    ARGS.dino_size = args.dino_size
    ARGS.memory_path = args.memory_path
    ARGS.scene_name = args.scene_name

    builder = PlaceGraphBuilder(ARGS)

    if NavEnv is None:
        print("[WARN] NavEnv not available in this standalone run. Only NPZ->graph works.")
        return

    env = NavEnv(ARGS, init_state=None, build_map=False)

    out_json = builder.build(
        env=env,
        random_move_num=args.random_move_num,
        turn_left_deg=args.turn_left_deg,
        lock_floor=args.lock_floor,
        floor_band_m=args.floor_band_m,
        save_rgb=args.save_rgb,
        rgb_stride=args.rgb_stride,
        rgb_format=args.rgb_format,
        save_depth=args.save_depth,
        depth_stride=args.depth_stride,
        depth_format=args.depth_format,
        depth_scale=args.depth_scale,
        cos_thresh=args.cos_thresh,
        yaw_gate_deg=args.yaw_gate_deg,
        tau_len=args.tau_len,
        knn_k=args.knn_k,
        r_max=args.r_max
    )
    print(f"[OK] Graph saved to {out_json}")


if __name__ == "__main__":
    main()
