# -*- coding: utf-8 -*-
"""
动作适配（连续 (v,w) → Habitat 动作）

提供两种模式:
1) DiscreteActionAdapter: 将 (v, ω, dt) 离散化为 Habitat 的 {TURN_LEFT/RIGHT, MOVE_FORWARD} 序列。
2) ContinuousActionAdapter: 直接返回速度命令（若你在 Habitat-Lab 中启用连续基座控制）。

注意:
- 离散化保留“残差累积”，避免每步四舍五入造成系统性偏差。
- 若 v 为负，简单策略是先转 180° 再前进（本文保留注释示例，默认 clamp 为 0）。
"""

from typing import List, Tuple, Dict, Optional
import math

class DiscreteActionAdapter:
    """
    把 (v, ω, dt) 映射为离散动作序列。
    参数:
        fwd_step: 单次 MOVE_FORWARD 的步长（米），需与 Habitat 配置一致
        turn_deg: 单次 TURN 的角度（度），需与 Habitat 配置一致
        v_max, w_max: 速度饱和
    """
    def __init__(self, fwd_step: float = 0.25, turn_deg: float = 30.0,
                 v_max: float = 0.8, w_max: float = 1.2) -> None:
        self.s = float(fwd_step)
        self.a = math.radians(turn_deg)
        self.v_max, self.w_max = float(v_max), float(w_max)
        self._yaw_residual = 0.0
        self._dist_residual = 0.0

    def reset(self) -> None:
        """重置残差累积（切换 episode 时建议调用）。"""
        self._yaw_residual = 0.0
        self._dist_residual = 0.0

    def from_vel(self, v: float, w: float, dt: float) -> List[str]:
        """
        将连续速度 (v, ω) 和时长 dt 转为离散动作序列。
        公式:
            d = clamp(v) * dt
            Δθ = clamp(ω) * dt
            n_fwd = round(d / s)
            n_rot = round(Δθ / a)

        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s), 左转为正
            dt: 控制时长 (s)

        Returns:
            seq: ["TURN_LEFT"/"TURN_RIGHT"/"MOVE_FORWARD", ...]
        """
        # 饱和
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))

        # 理想位移/转角 + 残差补偿
        d = v * dt + self._dist_residual
        dyaw = w * dt + self._yaw_residual

        # 将连续量四舍五入到离散步
        n_fwd = int(round(d / self.s))
        n_rot = int(round(dyaw / self.a))

        # 更新残差（下次补偿）
        self._dist_residual = d - n_fwd * self.s
        self._yaw_residual = dyaw - n_rot * self.a

        seq: List[str] = []

        # 先转向（经验上这样路径更平滑）
        if n_rot != 0:
            act = "TURN_LEFT" if n_rot > 0 else "TURN_RIGHT"
            for _ in range(abs(n_rot)):
                seq.append(act)

        # 再前进
        if n_fwd > 0:
            for _ in range(n_fwd):
                seq.append("MOVE_FORWARD")
        elif n_fwd < 0:
            # 可选策略（默认关闭）:
            # 1) seq += ["TURN_LEFT"]*18 or 36  # 180°
            # 2) seq += ["MOVE_FORWARD"] * abs(n_fwd)
            # 这里简单处理为不后退（留给上层控制器修正）
            pass

        return seq


class ContinuousActionAdapter:
    """
    连续控制适配器（若使用 Habitat-Lab 连续基座）。
    这里只是把 (v, ω, dt) 打包返回；实际发送需要你在上层调用
    habitat-lab 的相应 API（如 base_velocity）。
    """
    def __init__(self, v_max: float = 0.8, w_max: float = 1.2) -> None:
        self.v_max, self.w_max = float(v_max), float(w_max)

    def from_vel(self, v: float, w: float, dt: float) -> Dict[str, float]:
        """
        打包连续命令。

        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s)
            dt: 控制时长 (s)

        Returns:
            {"lin": v_clamped, "ang": w_clamped, "dt": dt}
        """
        v = max(-self.v_max, min(self.v_max, v))
        w = max(-self.w_max, min(self.w_max, w))
        return {"lin": v, "ang": w, "dt": float(dt)}
