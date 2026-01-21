# -*- coding: utf-8 -*-
"""
这个文件和wangbo_run_navdp_follow_path_continuous_total.py是一模一样的
这个文件的目的主要是提高运行的速度，加快跑出结果的速度
主要区别如下：
1. 关掉可视化, args.video_path = None
2. 不再适用env.get_metric()来获取指标，因为每次会计算SPL, topdownmap，太慢了
3. 我把env中的update给关掉了，后续可以打开
3. 关闭path plan中的路径规划更新，因为更新是为了env.get_metric计算的准，现在不用env.get_metric了就不需要这个了
"""


"""
NavDP × Graph：按图路径导航（ImageNav 场景）
- 每个 episode：
  1) 匹配 start/goal（goal 第一次确定后固定）
  2) 在 place-graph 上用 A* 求从“当前最近节点→start→…→goal”的节点路径
  3) 用 NavDP point-goal 逐段走（current → 下一个节点中心），滚动重规划
  4) 每隔 R 匹配一次当前位置所属/最近节点，若偏离则用“最近节点→goal”重算后续路径
  5) 成功：距 goal ≤ success_distance；失败：总步数 > max_total_steps
"""

import pandas as pd
import os, math, json, argparse, socket, sys
from typing import List, Tuple, Optional, Dict
import numpy as np
import imageio.v2 as imageio
import cv2
from PIL import Image
from env import get_objnav_env
from wangbo_localization import ImageNavGraphRobot
from habitat_sim.utils.common import quat_to_magnum
import magnum as mn
from scipy.spatial.transform import Rotation as R
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height, to_grid  
import torch
from tqdm import tqdm
import time 

from pathlib import Path
import rclpy
from rosbag_rgbd_load import RGBDOdomSource

from robot_visualization import GraphBGVideoWriter

sys.path.append("./third-party/NavDP")
from adapters_habitat.camera_adapter import intrinsics_from_hfov
from adapters_habitat.pose_adapter import to_navdp_pos, to_navdp_rot, yaw_from_R_navdp
from adapters_habitat.path_follower import cam_traj_to_world, PathFollowerDiscrete, HabitatController
from utils_tasks.client_utils import navigator_reset as _navdp_navigator_reset
from utils_tasks.client_utils import pointgoal_step as _navdp_pointgoal_step
from utils_tasks.client_utils import imagegoal_step as _navdp_imagegoal_step

sys.path.append("./third-party/Grounded-SAM-2")
from grounded_sam2_wrapper import GroundedSAM2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- 小工具 ----------

def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def _ensure_uint8_rgb3(arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0: arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:                arr = arr.astype(np.uint8)
    if arr.shape[-1] == 4: arr = arr[..., :3]
    return arr
    
def _to_uint8_rgb(img):
    """把任意 (H,W),(H,W,1),(H,W,3/4)、float/uint8 都转成 uint8 RGB3。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    # float -> uint8
    if arr.dtype != np.uint8:
        # 假设已是 0~1 或任意范围；clip 后缩放
        a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max - a_min < 1e-9:
            arr = np.zeros_like(arr, dtype=np.uint8)
        else:
            arr = ((arr - a_min) / (a_max - a_min) * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def center_pad_or_crop(img: np.ndarray, target_h: int, target_w: int, pad_value: int = 0) -> np.ndarray:
    """
    将图像“中心对齐”到指定大小：
    - 若当前尺寸小于 target：上下/左右等距补边（pad_value为边色，默认黑）
    - 若当前尺寸大于 target：上下/左右居中裁剪
    - 保持 H×W×3、uint8，不缩放
    """
    h, w = img.shape[:2]

    # ---- 先处理高度 ----
    if h < target_h:
        pad = target_h - h
        top = pad // 2
        bottom = pad - top
        img = np.pad(img,
                     ((top, bottom), (0, 0), (0, 0)),
                     mode="constant",
                     constant_values=((pad_value, pad_value), (0, 0), (0, 0)))
    elif h > target_h:
        off = (h - target_h) // 2
        img = img[off:off + target_h, :, :]

    # ---- 再处理宽度 ----
    h, w = img.shape[:2]
    if w < target_w:
        pad = target_w - w
        left = pad // 2
        right = pad - left
        img = np.pad(img,
                     ((0, 0), (left, right), (0, 0)),
                     mode="constant",
                     constant_values=((0, 0), (pad_value, pad_value), (0, 0)))
    elif w > target_w:
        off = (w - target_w) // 2
        img = img[:, off:off + target_w, :]

    return img

def _depth_to_rgb(depth):
    """depth -> uint8 RGB3, 自动归一化、处理 nan/inf。"""
    d = np.asarray(depth)
    if d.ndim == 3:  # 可能是 (H,W,1)
        d = d[..., 0]
    # 归一化
    finite = np.isfinite(d)
    if not finite.any():
        g = np.zeros_like(d, dtype=np.uint8)
    else:
        dmin, dmax = float(np.nanmin(d)), float(np.nanmax(d))
        g = ((d - dmin) / (dmax - dmin + 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
        g[~finite] = 0
    return np.stack([g, g, g], axis=-1)

def _resize_h(img, h):
    H, W = img.shape[:2]
    if H == h: 
        return img
    new_w = int(round(W * (h / H)))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

def load_goal_as_habitat_obs_rgb(goal_jpg_path: str):
    """
    Load goal image as Habitat-style RGB observation:
      - np.uint8
      - (H, W, 3)
      - RGB channel order
    No resizing, no env dependency.
    """
    img = Image.open(str(goal_jpg_path)).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb

def build_rgbpath_to_xyz_index(frames_meta_jsonl: str):
    idx = {}
    with open(frames_meta_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rgb_path = rec.get("rgb_path")
            pose = rec.get("pose", {}) or {}
            if rgb_path:
                idx[rgb_path] = (
                    float(pose.get("x", 0.0)),
                    float(pose.get("y", 0.0)),
                    float(pose.get("z", 0.0)),
                )
    return idx

def node_center(robot, nid: int) -> np.ndarray:
    return np.array(robot.graph.nodes[int(nid)].center, dtype=float)

def _quat_conj_xyzw(q):
    # q: (x, y, z, w)
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float32)

def _quat_mul_xyzw(a, b):
    # Hamilton product, both (x, y, z, w)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz
    ], dtype=np.float32)

def _quat_rotate_vec_xyzw(q, v):
    # Rotate vector v (3,) by quaternion q (x,y,z,w): v' = q * (v,0) * q_conj
    vq = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    return _quat_mul_xyzw(_quat_mul_xyzw(q, vq), _quat_conj_xyzw(q))[:3]

def hab_to_ros_xyz(world_xyz_hab):
    """
    Habitat(X右,Y上,Z后) -> ROS(X前,Y左,Z上)
    """
    xh, yh, zh = map(float, world_xyz_hab)
    return np.array([-zh, -xh, yh], dtype=np.float32)

def world_to_rel_dxdy_from_odom(
    robot_pos_odom_xyz,
    robot_quat_odom_xyzw,
    world_xyz_hab,
):
    """
    输入
    ----
    robot_pos_odom_xyz: (3,)
        来自 /odom 的 position，ROS 坐标系（通常 X前 Y左 Z上）
    robot_quat_odom_xyzw: (4,)
        来自 /odom 的 orientation，四元数顺序 (x,y,z,w)
        表示：base_link 相对于 odom/world 的姿态
    world_xyz_hab: (3,)
        目标点在 Habitat 世界系坐标（X右 Y上 Z后）

    输出
    ----
    dx_forward, dy_left
        dx > 0：前方；dy > 0：左侧 （均在机器人机体系下）
    """
    robot_pos = np.asarray(robot_pos_odom_xyz, dtype=np.float32).reshape(3,)
    q = np.asarray(robot_quat_odom_xyzw, dtype=np.float32).reshape(4,)

    # 1) 目标点 Habitat -> ROS world(odom)
    goal_ros = hab_to_ros_xyz(world_xyz_hab)

    # 2) 世界系位移（odom/world）
    d_world = goal_ros - robot_pos

    # 3) 把世界系位移旋回机体系：v_body = q^{-1} * v_world
    v_body = _quat_rotate_vec_xyzw(_quat_conj_xyzw(q), d_world)

    # 4) ROS 机体系下：X前 Y左
    dx_forward = float(v_body[0])
    dy_left    = float(v_body[1])
    return np.array([dx_forward, dy_left], dtype=np.float32)

def habitat_xyz_to_ros_xyz(hab_xyz):
    """
    Habitat (X右, Y上, Z后)
        -> ROS (X前, Y左, Z上)
    """
    xh, yh, zh = hab_xyz
    xr = -zh   # 前
    yr = -xh   # 左
    zr =  yh   # 上
    return np.array([xr, yr, zr], dtype=np.float32)

def rel_dxdy_to_world(cam_pos, cam_rot, dx_fwd: float, dy_left: float, up: float = 0.0):
    """
    将相对位移 [dx_fwd, dy_left, up]（dx=前+, dy=左+, up=上+）
    转成：
      1) 世界系位移向量 disp_world:  [dx_w, dy_w, dz_w]
      2) 世界系目标点   goal_world:  [x_w,  y_w,  z_w] = agent_pos + disp_world

    机体系轴向（Habitat常见）：X=右, Y=上, Z=后(→ -Z为前)。
    映射：v_agent = [-dy_left, up, -dx_fwd]
    """
    # 1) 相对位移在机体系下的向量
    v_agent = mn.Vector3(-float(dy_left), float(up), -float(dx_fwd))

    # 2) 四元数（world <- agent），把机体系向量旋到世界系
    v_world = cam_rot.transform_vector(v_agent)

    # 3) 输出：世界系位移 & 世界系目标点
    disp_world = np.array([float(v_world.x), float(v_world.y), float(v_world.z)], dtype=np.float32)
    goal_world = cam_pos + disp_world
    return goal_world



def world_xz_to_rot_px(env, x_world: float, z_world: float, td):
    H0, W0 = td["map"].shape  # 旋转之前的二值栅格尺寸 (rows, cols)
    # to_grid 的 realworld_x 对应世界坐标 z，realworld_y 对应世界坐标 x
    sim = getattr(env, "sim", getattr(env, "_sim", None))
    assert sim is not None and hasattr(sim, "pathfinder"), "env 没有 sim/pathfinder"
    r, c = to_grid(
        realworld_x=z_world,
        realworld_y=x_world,
        grid_resolution=(H0, W0),
        pathfinder=sim.pathfinder,
    )  # r=row, c=col

    if H0 > W0:
        # colorize 会先对原图 np.rot90(..., 1)（逆时针 90°）
        # 原 (r,c) -> 旋转后 (row' = W0-1-c, col' = r)
        # 我们需要 (x,y) = (col', row')
        x_px = float(r)
        y_px = float(W0 - 1 - c)
    else:
        # 未旋转，直接 (x,y)=(c,r)
        x_px = float(c)
        y_px = float(r)
    return (x_px, y_px)




# ---------- NavDP RPC 适配 ----------
def navdp_reset(intrinsic_K: np.ndarray, batch_size: int, stop_threshold: float, host: str, port: int):
    try:
        return _navdp_navigator_reset(intrinsic_K, batch_size=batch_size, stop_threshold=stop_threshold, host=host, port=port)
    except TypeError:
        return _navdp_navigator_reset(intrinsic_K, batch_size=batch_size, stop_threshold=stop_threshold, port=port)

def navdp_pointgoal(goal_nav_2d: np.ndarray, rgb: np.ndarray, depth: np.ndarray, host: str, port: int):
    try:
        return _navdp_pointgoal_step(goal_nav_2d, rgb, depth, host=host, port=port)
    except TypeError:
        return _navdp_pointgoal_step(goal_nav_2d, rgb, depth, port=port)

def navdp_imagegoal(goal_nav_3d: np.ndarray, rgb: np.ndarray, depth: np.ndarray, host: str, port: int):
    try:
        return _navdp_imagegoal_step(goal_nav_3d, rgb, depth, host=host, port=port)
    except TypeError:
        return _navdp_imagegoal_step(goal_nav_3d, rgb, depth, port=port)



# ---------- 主流程 ----------
def run_episode_with_graph(
    goal,
    goal_xyz: np.ndarray,
    rosbag_reader,
    robot: ImageNavGraphRobot,
    rpc_host: str = "127.0.0.1",
    rpc_port: int = 8888,
    hfov_deg: float = 90.0,
    success_distance: float = 1.0,
    re_localize_every: int = 30,                        # 每隔多少执行步做一次“视觉重定位+重规划”
    max_total_steps: int = 500,
    look_ahead_dist: float = 0.25,                   # pure pursuit中 look ahead的距离设定
    control_steps: int = 2,                      # 沿着同一段traj走多久
    video_path: Optional[str] = None,
    # === 🆕 新增卡住检测参数 ===
    stuck_check_steps: int = 5,       # 连续多少步判断一次卡住
    stuck_threshold: float = 0.1,     # 平均位移小于该阈值视为卡住（单位: m）
    min_relocalize_before_fallback: int = 2,    # ← 至少成功做过2次re-localize后才允许fallback
    pano_step_deg: float = 30.0,     # 每次转动的角度
    pano_rounds: int = 1,            # 旋转几圈（1 就够用）
) -> Dict:
    """
    返回：{"success": 0/1, "steps": N}

    关键点：
    - 阶段A（initial → start_center）和阶段B（start → ... → goal）统一为“沿着扩展路径的第一个 waypoint 开始走”，
      并且两阶段都每隔 re_localize_every 步做一次基于图像匹配的“当前位置节点”重定位，然后重算后续路径。
    - 始终用 NavDP point-goal 生成“短期轨迹”，用 PathFollowerDiscrete 量化为离散动作后执行。
    """


    # ---------- 0) 准备 ----------
    out = None
    while out is None:
        rclpy.spin_once(rosbag_reader, timeout_sec=0.01)
        out = rosbag_reader.get_latest()
    
    if out is None:
        print("Failed to get observation from rosbag.")

    t_ros, obs0, depth0, pos, quat = out
    print("Initial position from rosbag:", pos)
    start_xyz = np.array(pos, dtype=float)

    # 相机内参 -> 注册 NavDP（batch=1）
    H, W = obs0.shape[:2]
    K, _, _ = intrinsics_from_hfov(H, W, hfov_deg)
    try:
        navdp_reset(K, batch_size=1, stop_threshold=-10.0, host=rpc_host, port=rpc_port)
    except Exception as e:
        print(f"[WARN] navigator_reset 失败：{e}")

    start_image = obs0
    goal_image = goal
    goal_image = goal_image[None, ...]  

    def node_center(nid: int) -> np.ndarray:
        return np.array(robot.graph.nodes[int(nid)].center, dtype=float)


    # ---------- 1) 起终点匹配（goal 第一次确定后固定） ----------
    start_xyz, goal_xyz = start_xyz, goal_xyz
    start_top = robot.localize_obs_with_instance_real_robot(start_image)
    goal_top  = robot.localize_obs_with_instance_real_robot(goal_image[0])

    start_nid = int(start_top[0][0])
    goal_nid  = int(goal_top[0][0])

    start_center = node_center(start_nid)
    goal_center  = node_center(goal_nid)     # 成功判定固定点

    # ---------- 2) 初始全局路径（start → goal），并把“去 start”也纳入扩展路径 ----------
    A_path = robot.plan_waypoints_with_true_points(
        true_start_xyz=start_xyz,
        retrieved_start_node_id=start_nid,
        retrieved_goal_node_id=goal_nid,
        true_goal_xyz=goal_xyz,
    )
    path_nodes = A_path["node_path"]
    waypoints: List[np.ndarray] = [node_center(n) for n in path_nodes]

    # 把“节点ID→扩展路径下标”的映射，便于重定位后快速跳转到对应段
    id2wpi: Dict[int, int] = { start_nid: 0 }
    for j in range(1, len(path_nodes)):
        id2wpi[int(path_nodes[j])] = j  # 对应扩展路径下标


    ####### 可视化预处理

    #  需要写的mp4
    video_path = "/home/wangbo/codes/BSC-Nav/navdp_debug.mp4"   # 改成你想要的路径
    writer = GraphBGVideoWriter(
        graph=robot.graph,
        out_mp4_path=video_path,
        fps=10,
        frame_size=(1024, 1024),
        margin_m=2.0
    )

    writer.set_offset_by_start_alignment(robot_xyz0=start_xyz, start_center_hab_xyz=start_center)
    print("saving to:", video_path)

    # ---------- 3) 主循环：滚动重规划 + HabitatController 执行 ----------
    steps = 0
    success = False
    wp_idx = -1                      # 当前要去的扩展路径 waypoint 下标（0 是“去 start”的阶段A）
    since_last_reloc = 0

    predict_trj = []

    # === 🆕 新增：记录最近若干步的位置，用于判断卡住 ===
    recent_positions: List[np.ndarray] = []
    stuck_counter = 0        # 🆕 全局卡顿计数器
    max_stuck_events = 5     # 🆕 最大允许卡顿次数，超过则认为路径无效
    break_outer = False
    successful_relocalize = 0 # ← 成功重定位次数
    last_reloc_nid = start_nid      # 最近一次重定位到的节点
    same_reloc_count = 0       # 连续重定位“没变”的次数（可用于你后续策略）

    while steps < max_total_steps and (not success):
        # ---- 取观测（不执行动作） ----

        t0 = time.perf_counter()

        out = None
        spin_count = 0  # 🆕 记录 spin_once 调用次数
        """传入robot观测 rgbd"""
        while out is None:
            rclpy.spin_once(rosbag_reader, timeout_sec=0.01)  # 减少 timeout，更频繁地轮询
            out = rosbag_reader.get_latest()
            spin_count += 1
        
        t1 = time.perf_counter()


        if out is None:
            print("Failed to get observation from rosbag.")
            break

        t_ros, obs, depth, pos, quat = out

        current_xyz = np.array(pos, dtype=float)
        current_xyzw = np.array(quat, dtype=float)

        cur_xyz_vis = current_xyz.copy()
        cur_xyzw_vis = current_xyzw.copy()

        rgb = obs[None, ...]                            # (1,H,W,3)
        depth = depth[..., None] if depth.ndim==2 else depth  # (H,W,1)
        depth = depth[None, ...]                                           # (1,H,W,1)

        # ---- 设定当前 point-goal（扩展路径的第 wp_idx 个）----
        tgt_world = waypoints[wp_idx]                       # 世界系 [x,y,z]
        tgt_rel = world_to_rel_dxdy_from_odom(current_xyz, current_xyzw, tgt_world) # navdp接受[dx, dy], dx forward, dy left
        """获取当前位姿（仅用于变换；不是定位）"""
        tgt_rel = tgt_rel[None, ...] # 加一个batch纬度

        # print("tgt_rel:", tgt_rel)

        t2 = time.perf_counter()
        # ---- NavDP 规划为相机系轨迹 → NavDP世界平面 ----
        try:
            traj_cam, _cands, _vals = navdp_pointgoal(tgt_rel, rgb, depth, host=rpc_host, port=rpc_port)
            # traj_cam 也是rel pos, [dx, dy, dw]向前，向左，转角
        except Exception as e:
            print(f"[RPC] pointgoal_step 失败：{e}")
            break
        t3 = time.perf_counter()
        predict_trj.append(traj_cam)
        # print("traj_cam:", traj_cam)
        print(
            f"[Timing] "
            f"ros={ (t1-t0)*1000:.1f} ms | "
            f"prep={ (t2-t1)*1000:.1f} ms | "
            f"navdp={ (t3-t2)*1000:.1f} ms | "
            f"total={ (t3-t0)*1000:.1f} ms"
        )

        traj_cam_np = np.asarray(traj_cam, dtype=float).squeeze()
        if traj_cam_np.ndim == 1:
            traj_cam_np = traj_cam_np[None, :]
        
        # print("traj_cam_np:", traj_cam_np)

        writer.write_frame(
            current_xyz=cur_xyz_vis,
            current_quat_xyzw=cur_xyzw_vis,
            traj_cam=traj_cam_np,
            traj_use_dw=True,                 # 如果dw不靠谱就改False
            arrow_size_px=10,
            traj_thickness=2,
            astar_node_path=path_nodes,
            text=f"step={steps} wp={wp_idx}"
        )

        steps += 1
        # ---- Habitat 连续控制 ----
        # for i in range(control_steps):
        #     "更换成real robot的控制策略，根据预测轨迹行走"
        #     follower.control(current_pos, yaw, np.array(trajectory_points_world, dtype=np.float32), look_ahead_dist)
        #     steps += 1
        #     since_last_reloc += 1

        #     # 成功条件
        #     "获取当前位置与goal位置的相对距离"
        #     rclpy.spin_once(rosbag_reader, timeout_sec=0.1)
        #     t_ros, obs, depth, pos, quat = rosbag_reader.get_latest()
        #     cur_p = np.array(pos, dtype=np.float32)
        #     goal_ros = habitat_xyz_to_ros_xyz(goal_center)
        #     # if np.linalg.norm(cur_p[[0,1]] - goal_ros[[0,1]]) <= success_distance:
        #     if np.linalg.norm(cur_p[[0,1]] - goal_ros[[0,1]]) <= 1.0:
        #         success = True

        if success:
            break


        # ---- waypoint 达成就推进 ----
        "判断是否到达waypoint"
        # tgt_world_ros = habitat_xyz_to_ros_xyz(tgt_world)
        # if np.linalg.norm(cur_p[[0,1]] - tgt_world_ros[[0,1]]) <= 0.6:
        #     wp_idx += 1
        #     since_last_reloc = 0
        #     continue
    writer.close()
    print("Saved video to:", video_path)
    # --------------------- 匹配当前位置下正确的goal instance方向---------------

    "这个是也进行一个align"
    # 向左转一圈，每次90度，转4次

    # rgbs = []
    # rgbs.append(obs)  # 初始朝向
    # step_deg = 45.0
    
    # for _ in range(int(360 / step_deg)):
    #     # 执行旋转动作
    #     # follower.turn(pano_step_deg)
    #     steps += 1

    #     # 获取新的观测
    #     out = None
    #     while out is None:
    #         rclpy.spin_once(rosbag_reader, timeout_sec=0.1)
    #         t_ros, obs, depth, pos, quat = rosbag_reader.get_latest()
        
    #     if out is None:
    #         print("Failed to get observation from rosbag during panorama.")
    #         break

    #     rgbs.append(obs)


    # align_info = robot.align_to_goal_coarse_dinov2_real_robot(
    # obs_rgbs=rgbs,
    # goal_rgb=goal_image[0],   # 你的goal_image是(1,H,W,3)
    # step_deg=45.0,            # 推荐先用45°，够快且不易漏
    # prefer_left=True,
    # verbose=True,
    # )
    # round_steps, real_steps = align_info["steps_per_round"], align_info["real_turn"]
    # steps = steps + round_steps + real_steps

    # real_turn_direction = align_info["real_direction"]
    # print(f"Align to goal done. Real turn: {real_turn_direction}, steps: {real_steps}")

    # # 真机进行旋转
    # # .......
    # distance_to_goal = np.linalg.norm(cur_p[[0,1]] - goal_xyz[[0,1]]) # 计算当前位置与goal的距离
    # if distance_to_goal <= success_distance:
    #     success = True
    # else:
    #     success = False
    
    
    # --------------------- Imagegoal NavDP ---------------------
    while steps < max_total_steps and (not success):
        # ---- 取观测（不执行动作） ----
        "取机器人真实观测rgbd"
        out = None
        while out is None:
            rclpy.spin_once(rosbag_reader, timeout_sec=0.1)
            out = rosbag_reader.get_latest()
        
        if out is None:
            print("Failed to get observation from rosbag.")
            break

        t_ros, obs, depth, pos, quat = out

        rgb = obs[None, ...]                            # (1,H,W,3)
        depth = depth[..., None] if depth.ndim==2 else depth  # (H,W,1)
        depth = depth[None, ...]                                           # (1,H,W,1)

        # ---- NavDP 规划为相机系轨迹 → NavDP世界平面 ----
        try:
            traj_cam, _cands, _vals = navdp_imagegoal(goal_image, rgb, depth, host=rpc_host, port=rpc_port)
            # traj_cam 也是rel pos, [dx, dy, dw]向前，向左，转角
        except Exception as e:
            print(f"[RPC] pointgoal_step 失败：{e}")
            break
        predict_trj.append(traj_cam)
        trajectory_points_world = []

        # ---- Habitat 连续控制 ----

        # for i in range(control_steps):
        #     traj_length = np.linalg.norm(traj_cam[0][-1][:2])
        #     follower.control(current_pos, yaw, np.array(trajectory_points_world, dtype=np.float32), look_ahead_dist, traj_length)
        #     steps += 1

        #     # 成功条件
        #     "获取当前位置与goal位置的相对距离"
        #     rclpy.spin_once(rosbag_reader, timeout_sec=0.1)
        #     t_ros, obs, depth, pos, quat = rosbag_reader.get_latest()
        #     cur_p = np.array(pos, dtype=np.float32)
        #     # if np.linalg.norm(cur_p[[0,1]] - goal_xyz[[0,1]]) <= success_distance:
        #     if np.linalg.norm(cur_p[[0,1]] - goal_xyz[[0,1]]) <= 1.0:
        #         success = True

        #     if success:
        #         break

        if success:
            break


    # 计算实际路径长度
    actual_path_length = 0.0
    # for i in range(len(traj_walk) - 1):
    #     p0 = np.asarray(traj_walk[i],   dtype=float)
    #     p1 = np.asarray(traj_walk[i+1], dtype=float)
    #     actual_path_length += np.linalg.norm(p1[[0, 2]] - p0[[0, 2]])

    return {"success": int(success), "steps": int(steps), "path_length": float(actual_path_length)}, predict_trj, stuck_counter


HABITAT_ROOT_DIR = "third-party/habitat-lab"
# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser("Real Robot Nav")
    parser.add_argument("--success_distance", type=float, default=1.0)
    parser.add_argument("--max_total_steps", type=int, default=500)
    parser.add_argument("--re_localize_every", type=int, default=30)

    parser.add_argument("--stuck_check_steps", type=int, default=15)
    parser.add_argument("--stuck_threshold", type=int, default=0.1)

    # —— graph / memory ——（复用你的参数）
    parser.add_argument("--graph_json", type=str, default="memory/00810-CrMo8WxCyVb/place_graph_min1.0_radius0.5.json", help="path to place_graph.json")
    parser.add_argument("--explore_npz", type=str, default="memory/00810-CrMo8WxCyVb/explore_log.npz", help="path to explore_log.npz")
    parser.add_argument("--floor_json", type=str, default="memory/00810-CrMo8WxCyVb/floor_data.json", help="path to floor_data.json")
    parser.add_argument("--floor_idx", type=int, default=0, help="if load_single_floor, single floor idx")
    parser.add_argument("--semantic", type=bool, default=True)
    parser.add_argument("--min_dis", type=float, default=1.0, help='FPS min')
    parser.add_argument("--radius", type=float, default=0.5, help="node radius")

    # --- DINO/Device ---
    parser.add_argument("--dino_size", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--device", type=str, default="cuda")

    # —— NavDP RPC —— 
    parser.add_argument("--rpc_host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=8989)

    # —— controller ——
    parser.add_argument("--control_freq", type=float, default=2.0, help="in Hz")
    parser.add_argument("--max_vel", type=float, default=0.25, help="in m/s")
    parser.add_argument("--max_ang_vel", type=float, default=0.5, help="in rad/s")
    parser.add_argument("--look_ahead_dist", type=float, default=0.25, help="in m, used for pure pursuit control")
    parser.add_argument("--control_time", type=float, default=1.00, help="running time for each trajectory")

    # —— out —— 
    parser.add_argument("--out_dir", type=str, default="./out_graph_navdp_imagegoal_min1.0_radius0.5")
    parser.add_argument("--record_video", action="store_true", default=False, help="Enable video recording")
    args = parser.parse_args()

    args.predefined_class = ['seating', 'chest of drawers', 'bed', 'bathtub', 'clothes', 'toilet', 'stool', 'sofa', 'sink', 'tv monitor', 'picture', 'cushion', 'towel', 'shower', 'counter', 'fireplace', 'chair', 'table', 'gym equipment', 'cabinet', 'plant']
    preload_dinov2 = torch.hub.load('third-party/dinov2', args.dino_size, source='local').to('cuda')

    # prepare groundedsam2
    preload_gsam2 = GroundedSAM2(
        sam2_checkpoint = "third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        gdino_id = "third-party/Grounded-SAM-2/grounding-dino-tiny",
        default_box_threshold = 0.25,
        default_text_threshold = 0.25,
        device="cuda",
    )

    # === 🆕 场景遍历：对 SCENE_ID_MAP 中的每个场景独立跑一遍 ===
    MEMORY_ROOT = "/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/all_finetuned_navdp_goal_match_baseline_m1.0_r0.5/question_scene"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/trainednavdp_goal_match_baseline_m1.0_r0.5/question_scene"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/goal_match_baseline_m1.0_r0.5/question_scene"
    BASE_OUT = f"result/trainednavdp_goal_match_baseline_m{args.min_dis}_r{args.radius}"


    # === 🆕 动态切换本场景的输入/输出路径 & content_scenes ===
    scene_mem_dir = "/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/rgbd_export"
    args.graph_json  = os.path.join(scene_mem_dir, f"place_graph_min{args.min_dis}_radius{args.radius}.json")
    args.explore_npz = os.path.join(scene_mem_dir, "explore_log.npz")
    args.floor_json  = os.path.join(scene_mem_dir, "floor_data.json")
    # 为该 scene 单独的输出目录
    args.out_dir = os.path.join(MEMORY_ROOT, "result_nav", f"m{args.min_dis}_r{args.radius}")
    os.makedirs(args.out_dir, exist_ok=True)


    print("rplcy init.....")
    # 1) 机器人 & 环境
    robot = ImageNavGraphRobot(args,
                            graph_json=args.graph_json,
                            explore_npz=args.explore_npz,
                            preload_gsam=preload_gsam2, 
                            preload_dino=preload_dinov2)

    control_steps = int(args.control_time * args.control_freq)
    print(f"control fre: {args.control_freq}Hz, control_time: {args.control_time}s")
    print(f"agent will execute {control_steps} steps for each trajectory!")

    # ------- 从rosbag中读取goal和start观测 -------
    goal_path = scene_mem_dir + "/goal_image/instance_1/geo1.jpg"
    goal = load_goal_as_habitat_obs_rgb(goal_path)
    goal_xyz = np.array([4.8, 0, 0])

    print("rclpy init...")
    rclpy.init() # 接受数据
    src = RGBDOdomSource(
        rgb_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/aligned_depth_to_color/image_raw",
        odom_topic="/odom",
    )
    print("Rosbag_Reader Initialized.")

    ###  ----------推理并行走 ----------
    ret, pred_traj, stuck_num = run_episode_with_graph(
        goal, goal_xyz, 
        src,
        robot,
        rpc_host=args.rpc_host, rpc_port=args.rpc_port,
        success_distance=args.success_distance,
        re_localize_every=args.re_localize_every,
        max_total_steps=args.max_total_steps,
        look_ahead_dist=args.look_ahead_dist,
        control_steps = control_steps,
        stuck_check_steps=args.stuck_check_steps,
        stuck_threshold=args.stuck_threshold,
    )



if __name__ == "__main__":
    main()
