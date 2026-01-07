# -*- coding: utf-8 -*-
"""
相机内参适配（HFOV ↔ fx/fy, 以及投影校验）

用途:
- 给 navdp 的视觉模块提供 Isaac 风格 3×3 K 矩阵。
- 根据 HFOV 和分辨率推导 fx/fy，或已知 fx 反推 HFOV 以配置 Habitat 传感器。
"""

from typing import Tuple
import math
import numpy as np

def fx_from_hfov(hfov_rad: float, W: int) -> float:
    """
    由水平视场角和分辨率宽度计算 fx。

    Args:
        hfov_rad: 水平视场角（弧度）
        W: 图像宽度（像素）

    Returns:
        fx: 焦距（像素）
    """
    return (W / 2.0) / math.tan(hfov_rad / 2.0)

def fy_from_vfov(vfov_rad: float, H: int) -> float:
    """
    由垂直视场角和分辨率高度计算 fy。
    """
    return (H / 2.0) / math.tan(vfov_rad / 2.0)

def intrinsics_from_hfov(H: int, W: int, hfov_deg: float) -> Tuple[np.ndarray, float, float]:
    """
    根据 HFOV 构造 3×3 内参矩阵 K（Isaac 风格）。

    Args:
        H, W: 图像高宽
        hfov_deg: 水平视场角（度）

    Returns:
        K: (3,3) 内参矩阵
        hfov: 弧度
        vfov: 弧度
    """
    hfov = math.radians(hfov_deg)
    vfov = 2 * math.atan(math.tan(hfov / 2.0) * (H / float(W)))
    fx = fx_from_hfov(hfov, W)
    fy = fy_from_vfov(vfov, H)
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx],
                  [0,  fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K, hfov, vfov

def hfov_from_target_fx(fx: float, W: int) -> float:
    """
    已知目标 fx，反推 HFOV（度），可用于设置 Habitat 传感器 HFOV 与 Isaac 对齐。
    """
    hfov_rad = 2 * math.atan((W / 2.0) / fx)
    return math.degrees(hfov_rad)

def verify_projection(P_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    简单投影校验（不含畸变/外参）：将相机坐标系下 3D 点投影到像素坐标。
    常用于单元测试：检查 fx/fy/cx/cy 是否工作正常。

    Args:
        P_cam: shape (N,3), 相机坐标系下的 3D 点（Z>0）
        K: shape (3,3), 内参矩阵

    Returns:
        uv: shape (N,2), 像素坐标
    """
    X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    # 避免除零
    Z = np.clip(Z, 1e-6, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return np.stack([u, v], axis=-1)
