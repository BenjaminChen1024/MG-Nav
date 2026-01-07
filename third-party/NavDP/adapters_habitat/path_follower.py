# -*- coding: utf-8 -*-
"""
Path Follower (基于轨迹点的动作生成)
-----------------------------------
用途：将 navdp 输出的 *相机系轨迹* 转成 *世界系轨迹*(NavDP/Isaac坐标系)，
然后把世界系轨迹离散为 Habitat 的动作序列（或连续控制）。

核心接口：
1) cam_traj_to_world(): 相机系 -> 世界系（NavDP/Isaac坐标）
2) PathFollowerDiscrete: 根据当前位姿与世界系轨迹，生成 ["TURN_*", "MOVE_FORWARD"] 序列
3) PathFollowerContinuous: 可选，生成 (v, ω)（若你启用连续基座）

注意：
- 本文件只处理 *NavDP/Isaac 风格* 的坐标系。Habitat 的位姿需要先用 pose_adapter 做变换。
- 轨迹点通常是二维平面（x_forward, y_left），z 只用于对齐/可视化，不用于规划。
"""

from typing import List, Optional, Tuple
import math
import numpy as np


# =========================
# 1) 相机系 -> 世界系 变换
# =========================
def cam_traj_to_world(
    traj_cam_xy: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_rot_world: np.ndarray,
) -> np.ndarray:
    """
    将 navdp 返回的 *相机坐标系* 轨迹（二维: x_forward, y_left）映射到 *世界坐标系*(NavDP/Isaac坐标)。

    Args:
        traj_cam_xy: shape (N, 2) 或 (N, >=2)，每个点是 [x_cam_forward, y_cam_left]
                     注意：navdp 的相机系默认 x=前, y=左, 与世界系轴向一致，仅需旋转+平移
        cam_pos_world: shape (3,), 相机在世界系中的位置 p_world
        cam_rot_world: shape (3,3), 相机姿态在世界系中的旋转矩阵 R_world_from_cam

    Returns:
        traj_world_xy: shape (N, 2)，世界系下的二维轨迹点 [x_world, y_world]
    """
    if traj_cam_xy.ndim != 2 or traj_cam_xy.shape[1] < 2:
        raise ValueError("traj_cam_xy should be (N,2) at least.")
    N = traj_cam_xy.shape[0]
    # 扩展到3D局部点 [x, y, 0]
    local_3d = np.concatenate([traj_cam_xy[:, :2], np.zeros((N, 1))], axis=1)  # (N,3)
    # 世界系点 = 平移 + 旋转 * 局部点
    world_3d = cam_pos_world[None, :] + (cam_rot_world @ local_3d.T).T  # (N,3)
    return world_3d[:, :2]


# =========================
# 2) 离散动作 PathFollower
# =========================
class PathFollowerDiscrete:
    """
    将 *世界系轨迹*(NavDP/Isaac坐标) 转换为 Habitat 离散动作序列的路径跟随器。

    策略：
      - 对当前目标航点做两阶段动作：先转向(量化到 TURN_ANGLE)，再前进(量化到 FORWARD_STEP_SIZE)。
      - 到达阈值内（waypoint_tol）则切换到下一个航点。
      - 提供 yaw_tol，避免为很小的角度误差反复转向。
      - 每次调用仅生成有限步数（max_turn_steps_per_call, max_fwd_steps_per_call），便于交互式推进。

    坐标约定（全部 NavDP/Isaac 风格）：
      - 当前位置 pose: (x, y, yaw)，yaw=0 朝 x(前)，逆时针为正。
      - 轨迹 points: [[x1,y1], [x2,y2], ...]

    返回：
      - Habitat 动作字符串序列：["TURN_LEFT","TURN_RIGHT","MOVE_FORWARD",...]
    """

    def __init__(
        self,
        forward_step: float = 0.25,      # 与 Habitat 配置一致
        turn_angle_deg: float = 30.0,    # 与 Habitat 配置一致
        waypoint_tol: float = 0.20,      # 航点到达阈值（米）
        yaw_tol_deg: float = 12.0,        # 角度阈值（度）
        max_turn_steps_per_call: int = 6,
        max_fwd_steps_per_call: int = 6,
    ) -> None:
        self.s = float(forward_step)
        self.a = math.radians(turn_angle_deg)
        self.waypoint_tol = float(waypoint_tol)
        self.yaw_tol = math.radians(yaw_tol_deg)
        self.max_turn_steps = int(max_turn_steps_per_call)
        self.max_fwd_steps = int(max_fwd_steps_per_call)

        self._path: Optional[np.ndarray] = None  # (M,2)
        self._idx: int = 0                      # 当前目标索引
        self._done: bool = False

    def reset(self) -> None:
        """重置内部状态（切换 episode 时调用）"""
        self._path = None
        self._idx = 0
        self._done = False

    def set_path(self, world_path_xy: np.ndarray) -> None:
        """
        载入“世界系轨迹”。

        Args:
            world_path_xy: shape (M,2)，世界系轨迹
        """
        if world_path_xy.ndim != 2 or world_path_xy.shape[1] != 2:
            raise ValueError("world_path_xy must be (M,2).")
        self._path = world_path_xy.copy()
        self._idx = 0
        self._done = False

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度归一化到 (-pi, pi]"""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _next_waypoint(self) -> Optional[np.ndarray]:
        """返回当前目标航点（或 None 表示结束）"""
        if self._path is None:
            return None
        if self._idx >= len(self._path):
            return None
        return self._path[self._idx]

    def step(
        self,
        pose_navdp: Tuple[float, float, float],
        emit_stop_when_done: bool = True,
    ) -> List[str]:
        """
        生成一小段动作序列（不超过 max_*_per_call），逐步把 agent 推向当前航点。

        Args:
            pose_navdp: (x, y, yaw) in NavDP/Isaac 坐标
            emit_stop_when_done: 到达最后一个航点后是否发出 "STOP"

        Returns:
            actions: List[str]，Habitat 离散动作
        """
        actions: List[str] = []
        if self._done or self._path is None:
            return actions

        x, y, yaw = float(pose_navdp[0]), float(pose_navdp[1]), float(pose_navdp[2])

        # 若已到当前航点，推进索引
        wp = self._next_waypoint()
        if wp is None:
            self._done = True
            if emit_stop_when_done:
                actions.append("STOP")
            return actions

        # 循环：先确保朝向大致对齐，再向前推进；限制每次生成的步数
        turn_budget = self.max_turn_steps
        fwd_budget = self.max_fwd_steps

        # --- 1) 转向 ---
        # 目标角度 = 面向“当前航点”的方向
        dx, dy = wp[0] - x, wp[1] - y
        dist_to_wp = math.hypot(dx, dy)

        # 如果已经接近航点，切换到下一个；并在本次调用里继续处理下一个
        if dist_to_wp <= self.waypoint_tol:
            self._idx += 1
            # 递归式地继续处理后续航点（仅一次，避免太深递归）
            return self.step((x, y, yaw), emit_stop_when_done)

        desired = math.atan2(dy, dx)
        yaw_err = self._wrap_angle(desired - yaw)

        # 若角度误差大于阈值，则量化成 TURN 步
        if abs(yaw_err) > self.yaw_tol and turn_budget > 0:
            n_rot = int(round(yaw_err / self.a))
            # 限制每次最多转多少步，避免一次性转太多导致卡顿
            n_rot = max(-turn_budget, min(turn_budget, n_rot))
            if n_rot != 0:
                act = "TURN_LEFT" if n_rot > 0 else "TURN_RIGHT"
                for _ in range(abs(n_rot)):
                    actions.append(act)
                # 不在此处更新 yaw（由外层 runner 在执行后更新实际位姿）
                return actions

        # --- 2) 前进 ---
        if fwd_budget > 0:
            # 需要前进的步数（按步长离散）
            n_fwd = int(round(dist_to_wp / self.s))
            n_fwd = max(1, min(fwd_budget, n_fwd))  # 至少走一步，至多走预算
            for _ in range(n_fwd):
                actions.append("MOVE_FORWARD")
            return actions

        # 如果都没有生成动作（极少发生），返回空列表
        return actions

    def done(self) -> bool:
        """是否已经完成所有航点"""
        return self._done or (self._path is not None and self._idx >= len(self._path))


# =========================
# 3) 连续控制（可选）
# =========================
# abc
from abc import ABC, abstractmethod

# numpy
import numpy as np

# habitat
import habitat_sim
from habitat_sim.utils import common as utils


class Controller(ABC):

    @abstractmethod
    def control(self, pos, yaw, path):
        """
        Executes the control logic for the agent
        :param pos: pos in metric units
        :param path: path in metric units
        :return:
        """
        pass

    # @abstractmethod
    # def turn_left(self):
    #     pass


class HabitatController(Controller):
    def __init__(self, args, env):
        self.env = env
        self.sim = self.env.sim
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

        self.control_frequency = args.control_freq
        self.max_vel = args.max_vel
        self.max_ang_vel = args.max_ang_vel
        self.time_step = 1.0 / self.control_frequency

    def reset(self):
        self.vel_control.linear_velocity = np.zeros(3, dtype=np.float32)
        self.vel_control.angular_velocity = np.zeros(3, dtype=np.float32)
        
    def compute_angle_vel(self, yaw, dx, dy, time_step, max_ang_vel, max_vel):
        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        angular_velocity = np.array([0.0, np.clip(angle_diff / time_step, -max_ang_vel, max_ang_vel), 0.0])
        return angular_velocity


    # def compute_velocity(self, current_pos, next_pos, yaw, time_step, max_ang_vel, max_vel):
    #     dx = next_pos[0] - current_pos[0]
    #     dy = next_pos[1] - current_pos[1]
    #     desired_angle = np.arctan2(dy, dx)
    #     angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

    #     angular_velocity = np.array([0.0, np.clip(angle_diff / time_step, -max_ang_vel, max_ang_vel), 0.0])

    #     if abs(angle_diff) < 0.005:  # Increased threshold to allow for small corrections
    #         angular_velocity = np.array([0.0, 0.0, 0.0])
    #         speed = np.clip(np.linalg.norm([dx, dy]) / time_step, 0, max_vel)
    #         linear_velocity = np.array([0.0, 0.0, -speed])
    #     else:
    #         linear_velocity = np.array([0.0, 0.0, 0.0])

    #     return angular_velocity, linear_velocity

    def compute_velocity(self, current_pos, next_pos, yaw, time_step, max_ang_vel, max_vel):
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - yaw + np.pi) % (2 * np.pi) - np.pi

        # ---------- 角速度 ----------
        w = np.clip(angle_diff / time_step, -max_ang_vel, max_ang_vel)
        angular_velocity = np.array([0.0, w, 0.0])

        # ---------- 线速度 ----------
        # 根据角度偏差平滑调整线速度（同时转弯前进）
        # 当偏角=0 -> 全速；偏角=90° -> 约0速
        angle_gain = max(0.0, np.cos(angle_diff))  # 平滑抑制大角度前进
        raw_speed = np.linalg.norm([dx, dy]) / time_step
        speed = np.clip(raw_speed * angle_gain, 0, max_vel)
        linear_velocity = np.array([0.0, 0.0, -speed])

        return angular_velocity, linear_velocity

    def control(self, pos, yaw, path, look_ahead_dist=0.2, traj_length=100.0):
        """
        轨迹跟随控制逻辑：pure pursuit控制思想
        pos: 当前agent在世界坐标的 [x, z] 位置
        yaw: 当前朝向弧度
        path: 轨迹点数组 (N,2) 或 (N,3)，单位: 米
        look_ahead_dist: 预瞄距离（下一个目标点与当前点至少相隔多少距离）
        """

        if path is None or len(path) == 0:
            return np.zeros(3), np.zeros(3)

        # ---------- 1) 找最近的轨迹点 ----------
        path_2d = path[:, :2] if path.shape[1] > 2 else path
        dists = np.linalg.norm(path_2d - pos[:2], axis=1)
        idx_near = np.argmin(dists)

        # ---------- 2) 往前找一个 lookahead 目标 ----------
        goal_idx = idx_near
        for i in range(idx_near, len(path_2d)):
            if np.linalg.norm(path_2d[i] - pos[:2]) >= look_ahead_dist:
                goal_idx = i
                break
        goal_point = path_2d[goal_idx]

        # ---------- 3) 计算控制速度 ----------
        self.vel_control.angular_velocity, self.vel_control.linear_velocity = \
            self.compute_velocity(pos, goal_point, yaw, self.time_step, self.max_ang_vel, self.max_vel)

        # ---------- 4) 移动 agent ----------  
        agent_state = self.sim.get_agent(0).state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )

        # manually integrate the rigid state
        target_rigid_state = self.vel_control.integrate_transform(
            self.time_step, previous_rigid_state
        )

        # snap rigid state to navmesh and set state to object/sim
        # calls pathfinder.try_step or self.pathfinder.try_step_no_sliding
        end_pos = self.sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )

        # set the computed state
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(
            target_rigid_state.rotation
        )
        self.sim.get_agent(0).set_state(agent_state)
        self.sim.step_physics(self.time_step)

        # 使用step_physics不会让env自动更新metrics,需要手动更新一下
        obs = self.env.sim.get_sensor_observations()

        self.env._task.measurements.update_measures(task=self.env._task, episode=self.env._current_episode, observations=obs, action="MOVE_FORWARD")


