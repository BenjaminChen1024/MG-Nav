# -*- coding: utf-8 -*-
"""
坐标系/姿态适配（Habitat ↔ NavDP/Isaac 风格）

Habitat 世界系（右手）:
    x: 右, y: 上, z: 朝向观察者(即 "backward"，负z是前)
NavDP/Isaac 风格（右手）:
    x: 前, y: 左, z: 上

我们用一个常量变换矩阵 M_H2N 完成轴/符号重排:
    p_nav = [ -z_H, -x_H, +y_H ] = M_H2N @ p_hab
旋转同理:
    R_nav = M_H2N @ R_hab @ M_H2N.T
"""

from typing import Tuple
import numpy as np
import math

# Habitat -> NavDP 的线性映射矩阵（仅轴/符号重排，无尺度变换）
M_H2N = np.array(
    [
        [0.0, 0.0, -1.0],  # x_nav = -z_hab
        [-1.0, 0.0, 0.0],  # y_nav = -x_hab
        [0.0, 1.0, 0.0],   # z_nav = +y_hab
    ],
    dtype=float,
)

def to_navdp_pos(p_h: np.ndarray) -> np.ndarray:
    """
    将 Habitat 世界系下的 3D 位置，映射到 NavDP/Isaac 风格坐标系。

    Args:
        p_h: shape (3,), Habitat 世界坐标 [x, y, z]

    Returns:
        p_n: shape (3,), NavDP 世界坐标 [x_fwd, y_left, z_up]
    """
    return M_H2N @ p_h

def to_habitat_pos(p_n: np.ndarray) -> np.ndarray:
    """
    将 NavDP/Isaac 风格位置映射回 Habitat 世界系。

    Args:
        p_n: shape (3,), NavDP 世界坐标

    Returns:
        p_h: shape (3,), Habitat 世界坐标
    """
    return M_H2N.T @ p_n

def to_navdp_rot(R_h: np.ndarray) -> np.ndarray:
    """
    旋转矩阵从 Habitat → NavDP/Isaac。

    Args:
        R_h: shape (3,3), Habitat 旋转矩阵

    Returns:
        R_n: shape (3,3), NavDP 旋转矩阵
    """
    return M_H2N @ R_h @ M_H2N.T

def to_habitat_rot(R_n: np.ndarray) -> np.ndarray:
    """
    旋转矩阵从 NavDP/Isaac → Habitat。

    Args:
        R_n: shape (3,3), NavDP 旋转矩阵

    Returns:
        R_h: shape (3,3), Habitat 旋转矩阵
    """
    return M_H2N.T @ R_n @ M_H2N

def yaw_from_R_navdp(R_n: np.ndarray) -> float:
    """
    从 NavDP 风格旋转矩阵提取平面朝向（yaw, 弧度）。

    约定:
      - x_nav: 前，y_nav: 左
      - yaw = 0 表示朝 x_nav 正方向，逆时针为正

    常用公式:
      yaw = atan2(R[1,0], R[0,0])

    Args:
        R_n: shape (3,3), NavDP 旋转矩阵

    Returns:
        yaw: float, 弧度
    """
    return math.atan2(R_n[1, 0], R_n[0, 0])
