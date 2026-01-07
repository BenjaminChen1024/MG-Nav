# grounded_sam2_wrapper.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
from PIL import Image
import supervision as sv
from supervision.draw.color import ColorPalette

# ---- 来自 Grounded-SAM-2 与 GroundingDINO 的官方 API ----
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import pycocotools.mask as mask_util
import grounding_dino.groundingdino.datasets.transforms as T

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """把任意 (H,W,3/4) float/uint8 规范到 RGB uint8 HxWx3。"""
    arr = img
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))           # CHW -> HWC
    if arr.dtype != np.uint8:
        m = float(arr.max()) if arr.size else 1.0
        if m <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _norm_text_prompt(tp: Union[str, List[str]]) -> str:
    """
    GroundingDINO 要求：全部小写，且每个类名以 '.' 结尾，然后拼接成一句话。
    允许输入 "chair. sofa." 或 ["chair", "sofa"]，都会变成 "chair. sofa."
    """
    if isinstance(tp, str):
        s = tp.strip().lower()
        # 归一化为句子，以 '.' 结尾
        if not s.endswith('.'):
            s = s if s.endswith(' .') else (s + '.')
        return s
    else:
        toks = []
        for t in tp:
            tt = str(t).strip().lower()
            if not tt:
                continue
            if not tt.endswith('.'):
                tt += '.'
            toks.append(tt)
        return ' '.join(toks)

def load_image_from_array(rgb_uint8: np.ndarray):
    """
    完全等价于 load_image(image_path) 的输出格式：
      返回 (image_source[H,W,3] uint8, image_transformed[3,H,W] float32)
    变换：短边=800、最长边≤1333 的等比缩放 + ToTensor + Normalize
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_np = _ensure_rgb_uint8(rgb_uint8)
    pil = Image.fromarray(img_np)
    image_transformed, _ = transform(pil, None)  # -> CHW float32
    return img_np, image_transformed

class GroundedSAM2:
    """
    一个“可移植”的 Grounded-SAM-2 封装：
      - 初始化时需要传入 repo_root（指向你克隆的 Grounded-SAM-2 根目录）或四个相对路径；
      - detect() 支持传入 RGB ndarray 或图片路径；
      - 返回每个目标的 {label, score, bbox(xyxy), mask(bool HxW)}，可选是否返回 mask 与可视化。

    典型用法：
        gsam = GroundedSAM2(
            default_box_threshold = 0.35,
            default_text_threshold = 0.25,
            device="cuda",
        )
        dets = gsam.detect(rgb_or_path=rgb, text_prompt=["chair","sofa"])
    """

    def __init__(
        self,
        sam2_checkpoint: str = "/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        gdino_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        default_box_threshold: float = 0.35,
        default_text_threshold: float = 0.25,
    ):
        self.device = device
        self.default_box_threshold = float(default_box_threshold)
        self.default_text_threshold = float(default_text_threshold)

        # # build SAM2 image predictor
        self._sam2 = build_sam2(
            sam2_model_config,
            sam2_checkpoint,
            device=self.device,
        )
        self._sam2_predictor = SAM2ImagePredictor(self._sam2)

        # 构建 GroundingDINO
        self.GD_processor = AutoProcessor.from_pretrained(gdino_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_id).to(self.device)

        # ADD: 仅 Ampere+ GPU 开 TF32（对 matmul/cudnn 有提速，数值仍是 FP32 语义）
        if (self.device.startswith("cuda") and torch.cuda.is_available() and
            torch.cuda.get_device_properties(0).major >= 8):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def single_mask_to_rle(self, mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def rle_to_single_mask(self, rle):
        m = mask_util.decode(rle)  # (H, W, 1) 或 (H, W)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m  

    @torch.inference_mode()
    def detect(
        self,
        rgb: np.ndarray,
        text_prompt: Union[str, List[str]] = "chair. sofa. tv monitor.",
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Args:
            rgb: RGB(H,W,3) uint8
            text_prompt: str 或 list[str]，自动规整为 "xxx. yyy."
            box_threshold: threshold for box
            text_threshold: threshold for caption
        Returns:
            detections: List[dict]，每个字典至少包含：
              {
                "class_names": class_names, ## "table", "sofa"
                "input_boxes": [x1,y1,x2,y2],
                "masks": rles style,
                "label": [class_name, confidence], ## used for visualization
                "class_ids": int ## used for visualization
              }
        """
        box_th = float(self.default_box_threshold if box_threshold is None else box_threshold)
        txt_th = float(self.default_text_threshold if text_threshold is None else text_threshold)
        caption = _norm_text_prompt(text_prompt)

        # ---- 构造 GroundingDINO 的输入（尽量沿用官方流程）----
        if rgb.shape[-1] == 4: # 有时可能传进来的是rgba格式
            rgb = rgb[..., :3]
        pil = Image.fromarray(rgb)   
        self._sam2_predictor.set_image(np.array(pil))
        inputs = self.GD_processor(images=pil, text=caption, return_tensors="pt").to(self.device)

        # ADD: 仅在 CUDA 设备且 GPU 支持 bfloat16 时开启 autocast(bf16)
        use_bf16 = (self.device.startswith("cuda")
                    and torch.cuda.is_available()
                    and torch.cuda.is_bf16_supported())

        # CHANGE: 用 with 上下文包住 HF 模型前向；不用 .__enter__()
        with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16,
                            enabled=use_bf16):
            outputs = self.grounding_model(**inputs)

        results = self.GD_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_th,
            text_threshold=txt_th,
            target_sizes=[pil.size[::-1]]
        )

        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        # --- ① 空检测：直接早返回，不去调用 SAM2 ---
        num = int(input_boxes.shape[0])
        if num == 0:
            return {
                "class_names": [],
                "input_boxes": [],
                "masks": [],
                "labels": [],
                "class_ids": [],
            }

        masks, scores, logits = self._sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)


        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]
        
        mask_rles = [self.single_mask_to_rle(mask) for mask in masks]

        detects = {
            "class_names": class_names,
            "input_boxes": input_boxes,
            "masks": mask_rles,
            "labels": labels,
            "class_ids": class_ids
        }


        return detects

    @staticmethod
    def visualize(
        self,
        rgb: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """
        轻量可视化（不依赖 supervision），返回 BGR 图（方便 cv2.imwrite）。
        """
        # ---- 构造 GroundingDINO 的输入（尽量沿用官方流程）----
        if rgb.shape[-1] == 4: # 有时可能传进来的是rgba格式
            rgb = rgb[..., :3]

        # OpenCV 可视化用 BGR
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        input_boxes = detections["input_boxes"]
        mask_rles = detections["masks"]
        masks = [self.rle_to_single_mask(r) for r in mask_rles]
        masks = np.stack(masks, axis=0)
        class_ids = detections["class_ids"]
        labels = detections["labels"]

        # 可视化
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame_label = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame_mask = mask_annotator.annotate(scene=annotated_frame_label, detections=detections)

        return annotated_frame_label, annotated_frame_mask
