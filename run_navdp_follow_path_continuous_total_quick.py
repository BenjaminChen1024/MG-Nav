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

def set_floor_filter_from_json(floor_json_path: str, floor_idx: Optional[int] = None):
    """从 floor_data.json 读取当前楼层的 y 范围；允许加一点 pad 以抗数值误差。
    
    返回：
        floor_filter: (ymin, ymax) 或 None（单层场景时）
        num_floors: 楼层总数
    """
    skip = False
    assert os.path.exists(floor_json_path), f"floor json not found: {floor_json_path}"
    with open(floor_json_path, "r") as f:
        fd = json.load(f)
    
    ranges = fd.get("ranges_m") or []
    num_floors = int(fd.get("num_floors"))
    
    # 如果只有一层，返回 None 表示不过滤
    if num_floors <= 1:
        return None, num_floors
    
    # 多层场景，返回指定楼层的范围
    idx = int(fd.get("current_floor", 0)) if floor_idx is None else int(floor_idx)
    # assert 0 <= idx < num_floors, f"floor idx {idx} out of range"
    if idx >= num_floors:        
        print("floor_idx is out of range")
        return False, num_floors
    else:
        ymin, ymax = float(ranges[idx][0]), float(ranges[idx][1])
        floor_filter = (ymin, ymax)
    
    return floor_filter, num_floors

import cv2
import numpy as np

def make_vw_frame(rgb_im, depth_im, graph_vis, panel_h=900, step=None, traj_len=None, goal_distance=None, imagegoal_mode=False, waypoint=0, A_path=None):
    """
    拼接视频帧：
      布局：
        左侧：Graph
        右侧上方：RGB
        右侧下方：Depth
      底部增加 Step、Path 信息与 Relative Goal Pose
    返回 uint8 RGB3
    """
    # ---- 格式统一 ----
    rgb = _to_uint8_rgb(rgb_im)
    # dep = _depth_to_rgb(depth_im)
    dep = _to_uint8_rgb(depth_im)

    g = np.asarray(graph_vis)
    if g.ndim == 2:
        g = np.stack([g] * 3, axis=-1)
    if g.shape[-1] == 4:
        g = g[..., :3]
    g_rgb = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)

    # ---- 高度缩放 ----
    g_rgb = _resize_h(g_rgb, panel_h)

    # RGB / Depth 各自一半高度
    half_h = panel_h // 2
    rgb = _resize_h(rgb, half_h)
    dep = _resize_h(dep, half_h)

    # ---- 右半部分上下拼接 ----
    right = np.vstack([rgb, dep])

    # 对齐高度
    max_h = max(g_rgb.shape[0], right.shape[0])
    if g_rgb.shape[0] < max_h:
        pad = max_h - g_rgb.shape[0]
        g_rgb = cv2.copyMakeBorder(g_rgb, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    elif right.shape[0] < max_h:
        pad = max_h - right.shape[0]
        right = cv2.copyMakeBorder(right, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # ---- 横向拼接 [Graph | (RGB+Depth)] ----
    frame = np.hstack([g_rgb, right])

    # ---- 底部信息栏 ----
    info_h = 200 # 文字高度
    info_bar = np.full((info_h, frame.shape[1], 3), 255, dtype=np.uint8)

    # ---- 文本内容 ----
    texts = []
    if step is not None:
        texts.append(f"Step: {step}")
    if traj_len is not None:
        texts.append(f"Path length: {traj_len:.2f} m")
    if goal_distance is not None:
        texts.append(f"Relative Goal Pose: {goal_distance:.2f} m")
    if imagegoal_mode is True:
        texts.append(f"NavDP ImageGoal Mode")
    else:
        texts.append(f"NavDP PointGoal Mode, A* Path:{A_path}, Waypoint:{waypoint}")
    # ---- 绘制文字 ----
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.0, 2
    x, y_start = 30, int(info_h * 0.45)
    for i, txt in enumerate(texts):
        y = y_start + i * 35
        cv2.putText(info_bar, txt, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # ---- 拼接底栏 ----
    frame = np.vstack([frame, info_bar])

    return frame



def node_center(robot, nid: int) -> np.ndarray:
    return np.array(robot.graph.nodes[int(nid)].center, dtype=float)

def world_to_rel_dxdy(cam_pos, cam_rot, world_xyz):
    """
    把世界坐标下的 goal 点转换为相对 agent 的 2D 目标:
    返回 [dx, dy]，其中:
        dx > 0 表示目标在前方（forward）
        dy > 0 表示目标在左侧（left）

    假设机体系轴向为: X=右, Y=上, Z=后(即 -Z 为前) —— Habitat 默认常见约定。
    若你的机器人前向轴不同，调整文末两行符号即可。

    参数
    ----
    -------habitat-sim坐标系-------
    cam_pos 为世界系位置 (x, y, z), 
    cam_rot 为姿态四元数（可被 quat_to_magnum 识别）, 
    world_xyz: array-like, shape (3,)
        世界坐标系下的位置 (x, y, z), 

    返回
    ----
    np.ndarray, shape (2,)
        [dx, dy] （dx=forward, dy=left）
    """
    # 1) 世界系位移向量：从 agent 到 goal
    if world_xyz.size != 3:
        raise ValueError(f"world_xyz 必须是长度3的向量，当前为 {world_xyz.shape}")

    g_w = np.asarray(world_xyz, dtype=np.float32)               # [3]
    d_world = g_w - cam_pos                                              # [3]

    # 2) 四元数 -> Magnum（若已是 mn.Quaternion 则直接用）
    q = cam_rot if isinstance(cam_rot, mn.Quaternion) else quat_to_magnum(cam_rot)
    # 2) 取 agent 姿态的“逆旋转”（用共轭四元数）把向量旋回机体系
    v_world = mn.Vector3(float(d_world[0]), float(d_world[1]), float(d_world[2]))
    v_agent = q.conjugated().transform_vector(v_world)  # [x_right, y_up, z_back]

    # 3) 映射到你要的定义：dx=前(+), dy=左(+)
    dx_forward = -float(v_agent[2])   # 前(+) = -Z
    dy_left    = -float(v_agent[0])   # 左(+) = -X

    return np.array([dx_forward, dy_left], dtype=np.float32)

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


def _draw_dot(img, p, color, r=4, thickness=-1):
    cv2.circle(img, (int(p[0]), int(p[1])), r, color, thickness, lineType=cv2.LINE_AA)

def _draw_text(img, p, text, color=(255, 255, 255)):
    cv2.putText(img, text, (int(p[0]) + 4, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

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

def _prepare_canvas_and_nodes(env, graph, canvas_h: int):
    """生成底图(vis)、等比缩放系数(scale)、节点像素(未缩放->再缩放)、id->idx映射"""
    metrics = env.get_metrics() or {}
    td = metrics.get("top_down_map", None)

    # 1) 生成最终可视化底图（BGR），内部若 H0>W0 会先 rot90 再缩放到 canvas_h
    vis = colorize_draw_agent_and_fit_to_height(td, canvas_h)  # BGR
    H_vis = vis.shape[0]

    # 2) 旋转前尺寸
    H0, W0 = td["map"].shape[:2]
    rotated = (H0 > W0)          # colorize 在 H>W 时做了 np.rot90(map, 1)

    # 3) 旋转后的“基准高度”（未缩放）
    H_base = W0 if rotated else H0

    # 4) 等比缩放系数（单一 scale）
    scale = H_vis / float(H_base)

    # 5) id->idx
    id2idx = {n.id: i for i, n in enumerate(graph.nodes)}

    # 6) 节点像素（先算旋转后的原始像素，再统一乘 scale）
    node_px = []
    for n in graph.nodes:
        x_px, y_px = world_xz_to_rot_px(env, n.center[0], n.center[2], td)  # 已考虑旋转
        node_px.append((x_px, y_px))

    return vis, td, scale, node_px, id2idx


def visualize_path_on_graph(
    env,
    graph,
    node_path: List[int],
    show_nodes: bool = True,
    show_edges: bool = True,
    node_radius: int = 2,
    path_thickness: int = 3,
    canvas_h: int = 1024,
    predicted_traj_world=None,
    executed_world_path=None,          
):
    vis, td, scale, node_px, id2idx = _prepare_canvas_and_nodes(env, graph, canvas_h)
    def up(p): return (p[0] * scale, p[1] * scale)

    # 底图：节点/边
    if show_nodes:
        for p in node_px:
            _draw_dot(vis, up(p), color=(180, 220, 255), r=node_radius)
    if show_edges:
        for uid, nbrs in graph.adj.items():
            ui = id2idx.get(uid, None)
            if ui is None:  continue
            pu = up(node_px[ui])
            for vid, _dist in nbrs:
                if vid <= uid:
                    continue
                vi = id2idx.get(vid, None)
                if vi is None:  continue
                pv = up(node_px[vi])
                cv2.line(vis, (int(pu[0]), int(pu[1])), (int(pv[0]), int(pv[1])),
                        color=(80,160,80), thickness=1, lineType=cv2.LINE_AA)

    # 叠加 图上路径
    if node_path and len(node_path) >= 2:
        for i in range(len(node_path)-1):
            a_id, b_id = node_path[i], node_path[i+1]
            if a_id not in id2idx or b_id not in id2idx:
                continue
            a = up(node_px[id2idx[a_id]])
            b = up(node_px[id2idx[b_id]])
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                    color=(50, 50, 255), thickness=path_thickness, lineType=cv2.LINE_AA)
        # 标注端点（图上）
        a0 = up(node_px[id2idx[node_path[0]]])
        an = up(node_px[id2idx[node_path[-1]]])
        _draw_dot(vis, a0, color=(0, 255, 255), r=7)
        _draw_text(vis, a0, f"PATH_START:{node_path[0]}", (0, 255, 255))
        _draw_dot(vis, an, color=(255, 255, 0), r=7)
        _draw_text(vis, an, f"PATH_GOAL:{node_path[-1]}", (255, 255, 0))

    # ---------- 新增：预测轨迹（洋红） ----------
    if predicted_traj_world is not None:
        pred = np.asarray(predicted_traj_world)
        # 当前传入的是[x,y,z], 前 左 上 ——> 对应到 habitat里面应该是 右上后 [-y, z, -x]
        # 支持 (N,2 [x,z]) 或 (N,3 [x,y,z])
        xs, zs = -pred[:, 1], -pred[:, 0]
        pts = [world_xz_to_rot_px(env, float(x), float(z), td) for x, z in zip(xs, zs)]
        pts = [(int(px*scale), int(py*scale)) for px, py in pts]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(vis, a, b, color=(200, 60, 200), thickness=path_thickness+3, lineType=cv2.LINE_AA)

    # # ---------- 新增：真实执行轨迹（天蓝） ----------
    # if executed_world_path is not None:
    #     exe = np.asarray(executed_world_path)
    #     if exe.shape[1] == 2:
    #         xs, zs = exe[:, 0], exe[:, 1]
    #     else:
    #         xs, zs = exe[:, 0], exe[:, 2]
    #     pts = [world_xz_to_rot_px(env, float(x), float(z), td) for x, z in zip(xs, zs)]
    #     pts = [(int(px*scale), int(py*scale)) for px, py in pts]

    #     # 1️⃣ 绘制执行轨迹线
    #     for a, b in zip(pts[:-1], pts[1:]):
    #         cv2.line(vis, a, b, color=(50, 120, 250), thickness=path_thickness+3, lineType=cv2.LINE_AA)

    #     # 2️⃣ 绘制方向箭头（最后一段方向）
    #     if len(pts) >= 2:
    #         p_tail, p_head = pts[-2], pts[-1]
    #         cv2.arrowedLine(vis, p_tail, p_head, color=(255, 255, 255),
    #                         thickness=7, tipLength=0.7, line_type=cv2.LINE_AA)

    return vis
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
    obs0,
    env,
    robot: ImageNavGraphRobot,
    follower: HabitatController,                      # ← 传入已配置好的 PathFollowerDiscrete
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

    # 相机内参 -> 注册 NavDP（batch=1）
    H, W = obs0["rgb"].shape[:2]
    K, _, _ = intrinsics_from_hfov(H, W, hfov_deg)
    try:
        navdp_reset(K, batch_size=1, stop_threshold=-10.0, host=rpc_host, port=rpc_port)
    except Exception as e:
        print(f"[WARN] navigator_reset 失败：{e}")

    goal_image = obs0["instance_imagegoal"] 
    goal_image = goal_image[None, ...]  

    gola_im_visualization = _ensure_uint8_rgb3(obs0["instance_imagegoal"])
    gola_im_vis = center_pad_or_crop(gola_im_visualization, target_h=H, target_w=W)

    def node_center(nid: int) -> np.ndarray:
        return np.array(robot.graph.nodes[int(nid)].center, dtype=float)


    # ---------- 1) 起终点匹配（goal 第一次确定后固定） ----------
    if getattr(robot.args, "semantic", True):
        start_xyz, goal_xyz, start_top, goal_top = robot.localize_start_goal_from_obs_with_instance(obs0, env)
    else:
        start_xyz, goal_xyz, start_top, goal_top = robot.localize_start_goal_from_obs(obs0, env)
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

    def _panorama_relocalize(env, robot, pano_step_deg=30.0, pano_rounds=1, last_reloc_nid=None):
        """
        原地旋转若干小步，边转边重定位；一旦估计节点发生变化，立刻返回该节点ID。
        返回：(new_nid 或 None, 新的 path_nodes/waypoints/id2wpi（如需重铺时）)
        """
        # 用真实的 TURN_ANGLE 来计算一圈步数，保证能转满
        try:
            turn_angle = float(env.config.SIMULATOR.TURN_ANGLE)
        except Exception:
            turn_angle = float(pano_step_deg)

        steps_per_round = max(1, int(round(360.0 / max(1e-3, turn_angle))))
        total_turns = steps_per_round * max(1, int(pano_rounds))
        for _ in range(total_turns):
            # 原地左转一次（SPL 不受影响）
            env.step({"action": "turn_left"})
            # 立刻重定位一次
            obs_loc = env.sim.get_sensor_observations(0)
            try:
                if getattr(robot.args, "semantic", True):
                    cur_top = robot.localize_obs_with_instance(obs_loc, env)
                else:
                    cur_top = robot.localize_obs(obs_loc, env)
                cur_nid_est = int(cur_top[0][0])
            except Exception:
                cur_nid_est = None

            if cur_nid_est is None:
                continue
    
            if (last_reloc_nid is None) or (cur_nid_est != last_reloc_nid):
                return cur_nid_est
        return None

    # ---------- 3) 主循环：滚动重规划 + HabitatController 执行 ----------
    steps = 0
    success = False
    wp_idx = 0                      # 当前要去的扩展路径 waypoint 下标（0 是“去 start”的阶段A）
    since_last_reloc = 0

    vw = imageio.get_writer(video_path, fps=10) if video_path else None
    traj_walk = []
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
        obs = env.sim.get_sensor_observations(0)
        rgb = obs["rgb"];  rgb = rgb[None, ...]                            # (1,H,W,3)
        depth = obs["depth"]; depth = depth[..., None] if depth.ndim==2 else depth  # (H,W,1)
        depth = depth[None, ...]                                           # (1,H,W,1)


        # ---- 当前位姿（仅用于变换；不是定位）----
        st = env.sim.get_agent_state(0)
        cam_rot = quat_to_magnum(st.rotation)
        cam_pos = np.array(st.position, dtype=float)

        traj_walk.append([float(st.position[0]), float(st.position[1]), float(st.position[2])])

        # ---- 设定当前 point-goal（扩展路径的第 wp_idx 个）----
        tgt_world = waypoints[wp_idx]                       # 世界系 [x,y,z]
        tgt_rel = world_to_rel_dxdy(cam_pos, cam_rot, tgt_world) # navdp接受[dx, dy], dx forward, dy left
        tgt_rel = tgt_rel[None, ...] # 加一个batch纬度

        # ---- NavDP 规划为相机系轨迹 → NavDP世界平面 ----
        try:
            traj_cam, _cands, _vals = navdp_pointgoal(tgt_rel, rgb, depth, host=rpc_host, port=rpc_port)
            # traj_cam 也是rel pos, [dx, dy, dw]向前，向左，转角
        except Exception as e:
            print(f"[RPC] pointgoal_step 失败：{e}")
            break
        predict_trj.append(traj_cam)
        trajectory_points_world = []
        for i, point in enumerate(traj_cam[0]):
            if i < 0:
                continue
            point_habitat = rel_dxdy_to_world(cam_pos, cam_rot, point[0], point[1], 0.0)
            point_world = np.array([-point_habitat[2], -point_habitat[0], point_habitat[1]]) # 变成 正常世界坐标系, x前 y左 z上
            trajectory_points_world.append(point_world)    

        # ---- Habitat 连续控制 ----
        orientation = st.rotation
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        # r to euler
        pitch, yaw, roll = r.as_euler("yxz")
        # pitch is actually around z
        # orientation is pitch!
        yaw = pitch
        current_pos = np.array([-st.position[2], -st.position[0], st.position[1]]) # 变成 正常世界坐标系, x前 y左 z上

        for i in range(control_steps):
            follower.control(current_pos, yaw, np.array(trajectory_points_world, dtype=np.float32), look_ahead_dist)
            steps += 1
            since_last_reloc += 1

            # 成功条件
            cur_p = np.array(env.sim.get_agent_state(0).position, dtype=float)
            # if np.linalg.norm(cur_p[[0,2]] - goal_center[[0,2]]) <= success_distance:
            if np.linalg.norm(cur_p[[0,2]] - goal_center[[0,2]]) <= 0.6:
                success = True
                

            traj_walk.append([cur_p[0], cur_p[1], cur_p[2]])
            last_obs = env.sim.get_sensor_observations(0)


            # 录像（可选）
            if vw is not None:
                path_graph_vis = visualize_path_on_graph(
                    env,
                    graph = robot.graph,
                    node_path=path_nodes,
                    show_nodes= True,
                    show_edges= True,
                    path_thickness=2,
                    node_radius=10,
                    canvas_h=1024,
                    predicted_traj_world=trajectory_points_world,
                    executed_world_path=traj_walk,
                )

                rgb_im = _ensure_uint8_rgb3(last_obs["rgb"])

                # depth_im = last_obs["depth"]
                # if depth_im.ndim == 3:
                #     depth_im = depth_im[..., 0]  # 去掉多余通道
                # depth_im = np.nan_to_num(depth_im)  # 去掉 NaN
                # depth_im = np.clip(depth_im / depth_im.max(), 0, 1)  # 归一化到 0-1
                # depth_im = (depth_im * 255).astype(np.uint8)
                # depth_im = np.stack([depth_im] * 3, axis=-1)  # 变成 RGB3
                frame = make_vw_frame(rgb_im, gola_im_vis, path_graph_vis, panel_h=900, step=steps, 
                                        traj_len=np.linalg.norm(traj_cam[0][-1][:2]), 
                                        goal_distance=env.get_metrics()['distance_to_goal'],
                                        waypoint=wp_idx,
                                        A_path=waypoints
                                        )
                vw.append_data(frame)
            
            # === 🆕 新增：记录最近位置 & 判断卡住 ===
            recent_positions.append(cur_p.copy())
            if len(recent_positions) > stuck_check_steps:
                recent_positions.pop(0)  # 只保留最近 K 个位置

                # 计算这些位置之间的最大位移
                max_disp = max(np.linalg.norm(recent_positions[-1][[0,2]] - p[[0,2]]) for p in recent_positions[:-1])
                if max_disp < stuck_threshold:
                    # 1) 先常规重定位
                    try:
                        if getattr(robot.args, "semantic", True):
                            cur_top = robot.localize_obs_with_instance(last_obs, env)
                        else:
                            cur_top = robot.localize_obs(last_obs, env)
                        cur_nid_est = int(cur_top[0][0])
                        successful_relocalize += 1
                    except Exception:
                        cur_nid_est = None

                    # 2) 若重定位结果“没变”（= 仍是最近一次的 nid，或干脆等于 start_nid），
                    #    则触发“原地旋转→再重定位”兜底；只要不变就旋转，直到有变化或转满 pano_rounds 圈
                    unchanged = (cur_nid_est is not None) and (last_reloc_nid is not None) and (cur_nid_est == last_reloc_nid)
                    if cur_nid_est is None or unchanged:
                        new_nid = _panorama_relocalize(
                            env, robot,
                            pano_step_deg=pano_step_deg,   # 你已有的参数
                            pano_rounds=1,                  # 每次只转一圈；想更细可以把 step_deg 调小
                            last_reloc_nid=last_reloc_nid
                        )

                        if new_nid is not None:
                            cur_nid_est = new_nid

                    if cur_nid_est is not None:
                        if cur_nid_est in id2wpi:
                            # 在原路径上：最多跳过当前这个 waypoint（+1）
                            SKIP_AHEAD_K = 2   # 想跳两格就设2，想完全跳到定位节点就设0
                            target_idx = id2wpi[cur_nid_est] + SKIP_AHEAD_K
                            wp_idx = min(max(wp_idx, target_idx), len(waypoints) - 1)
                            #wp_idx = max(wp_idx, min(id2wpi[cur_nid_est] + 1, len(waypoints) - 1))
                        else:
                            # 偏离：从估计节点重铺到 goal
                            new_path = robot.plan_waypoints_with_true_points(
                                true_start_xyz=node_center(cur_nid_est),
                                retrieved_start_node_id=cur_nid_est,
                                retrieved_goal_node_id=goal_nid,
                                true_goal_xyz=goal_center,
                            )
                            path_nodes = new_path["node_path"]
                            waypoints  = [node_center(n) for n in path_nodes]
                            id2wpi     = { int(path_nodes[j]): j for j in range(len(path_nodes)) }
                            # wp_idx     = min(1, len(waypoints)-1)
                            wp_idx     = min(0, len(waypoints)-1)

                    stuck_counter += 1
                    recent_positions.clear()
                    since_last_reloc = 0
                    print(f"[INFO] Stuck #{stuck_counter}: relocalize & (re)plan; wp_idx={wp_idx}")    

                    # 只有达到两项条件才允许 fallback：
                    # 1) 卡住次数到上限；2) 至少成功re-localize过 min_relocalize_before_fallback 次
                    # 🆕 如果连续多次卡住，则直接跳出整个阶段进入 imagegoal 模式
                    if (stuck_counter >= max_stuck_events) and (successful_relocalize >= min_relocalize_before_fallback):
                        print(f"[WARN] Agent repeatedly stuck ({stuck_counter} times) → fallback to imagegoal mode")
                        success = False
                        break_outer = True
                        break  # 跳出 control_steps 循环
                    break  # 跳出当前控制循环，进入下一个 waypoint

        if break_outer:
            print("[INFO] Exiting graph-navigation stage due to repeated stuck events.")
            break

        if success:
            break


        # ---- waypoint 达成就推进 ----
        if np.linalg.norm(cur_p[[0,2]] - tgt_world[[0,2]]) <= 0.6:
            wp_idx += 1
            since_last_reloc = 0
            continue

        # ---- 周期性“视觉重定位+重规划”（阶段A/阶段B都生效）----
        if since_last_reloc >= re_localize_every and not success and wp_idx < len(waypoints):
            since_last_reloc = 0

            last_obs = env.sim.get_sensor_observations(0)  # ← 这里强制刷新
            try:
                if getattr(robot.args, "semantic", True):
                    cur_top = robot.localize_obs_with_instance(last_obs, env)
                else:
                    cur_top = robot.localize_obs(last_obs, env)
                # 需要你在 robot 里实现/提供该接口；若暂时没有可先跳过
                cur_nid_est = cur_top[0][0]
            except Exception:
                cur_nid_est = None

            if cur_nid_est is not None:
                cur_nid_est = int(cur_nid_est)
                # a) 若估计节点在“start→goal”的节点路径里，就把 wp_idx 跳转到该节点后的下一个 waypoint
                if cur_nid_est in id2wpi:
                    # special: 若是 start_nid，则最少跳到 1（开始走 start→next）
                    jump_to = max(1, id2wpi[cur_nid_est])    # id2wpi[start_nid]=0
                    if jump_to < len(waypoints):
                        wp_idx = max(wp_idx, jump_to)
                else:
                    # b) 偏离原路径：直接用“当前估计节点 → goal”重算后半段路径
                    new_path = robot.plan_waypoints_with_true_points(
                        true_start_xyz=node_center(cur_nid_est),
                        retrieved_start_node_id=cur_nid_est,
                        retrieved_goal_node_id=goal_nid,
                        true_goal_xyz=goal_center,
                    )
                    #更新path_nodes，便于可视化
                    path_nodes = new_path["node_path"]

                    waypoints = [node_center(n) for n in path_nodes]  # 注意：此时不再需要“去 start”的特殊 0 号点
                    # 重建映射 & 从“当前节点后的第一个”开始
                    id2wpi = { int(path_nodes[j]): j for j in range(len(path_nodes)) }
                    wp_idx = min(1, len(waypoints)-1)

    break_outer = False

    # --------------------- 匹配当前位置下正确的goal instance方向---------------
    align_info = robot.align_to_goal_coarse_dinov2(
    env=env,
    goal_rgb=goal_image[0],   # 你的goal_image是(1,H,W,3)
    step_deg=45.0,            # 推荐先用45°，够快且不易漏
    prefer_left=True,
    verbose=True,
    )
    round_steps, real_steps = align_info["steps_per_round"], align_info["real_turn"]
    steps = steps + round_steps + real_steps
    if env.get_metrics()['distance_to_goal'] <= success_distance:
        success = True
    else:
        success = False
    
    # 录像（可选）
    last_obs = env.sim.get_sensor_observations()
    if vw is not None:
        path_graph_vis = visualize_path_on_graph(
            env,
            graph = robot.graph,
            node_path=path_nodes,
            show_nodes= True,
            show_edges= True,
            path_thickness=2,
            node_radius=10,
            canvas_h=1024,
            predicted_traj_world=trajectory_points_world,
            executed_world_path=traj_walk,
        )

        rgb_im = _ensure_uint8_rgb3(last_obs["rgb"])

        # depth_im = last_obs["depth"]
        # if depth_im.ndim == 3:
        #     depth_im = depth_im[..., 0]  # 去掉多余通道
        # depth_im = np.nan_to_num(depth_im)  # 去掉 NaN
        # depth_im = np.clip(depth_im / depth_im.max(), 0, 1)  # 归一化到 0-1
        # depth_im = (depth_im * 255).astype(np.uint8)
        # depth_im = np.stack([depth_im] * 3, axis=-1)  # 变成 RGB3

        frame = make_vw_frame(rgb_im, gola_im_vis, path_graph_vis, panel_h=900, step=steps, 
                                traj_len=0.0, 
                                goal_distance=env.get_metrics()['distance_to_goal'],
                                imagegoal_mode = False)
        vw.append_data(frame)
    # --------------------- Imagegoal NavDP ---------------------
    while steps < max_total_steps and (not success):
        # ---- 取观测（不执行动作） ----
        obs = env.sim.get_sensor_observations(0)
        rgb = obs["rgb"];  rgb = rgb[None, ...]                            # (1,H,W,3)
        depth = obs["depth"]; depth = depth[..., None] if depth.ndim==2 else depth  # (H,W,1)
        depth = depth[None, ...]                                           # (1,H,W,1)

        # ---- 当前位姿（仅用于变换；不是定位）----
        st = env.sim.get_agent_state(0)
        cam_rot = quat_to_magnum(st.rotation)
        cam_pos = np.array(st.position, dtype=float)

        # ---- NavDP 规划为相机系轨迹 → NavDP世界平面 ----
        try:
            traj_cam, _cands, _vals = navdp_imagegoal(goal_image, rgb, depth, host=rpc_host, port=rpc_port)
            # traj_cam 也是rel pos, [dx, dy, dw]向前，向左，转角
        except Exception as e:
            print(f"[RPC] pointgoal_step 失败：{e}")
            break
        predict_trj.append(traj_cam)
        trajectory_points_world = []
        for i, point in enumerate(traj_cam[0]):
            if i < 0:
                continue
            point_habitat = rel_dxdy_to_world(cam_pos, cam_rot, point[0], point[1], 0.0)
            point_world = np.array([-point_habitat[2], -point_habitat[0], point_habitat[1]]) # 变成 正常世界坐标系, x前 y左 z上
            trajectory_points_world.append(point_world)    

        # ---- Habitat 连续控制 ----
        orientation = st.rotation
        r = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
        # r to euler
        pitch, yaw, roll = r.as_euler("yxz")
        # pitch is actually around z
        # orientation is pitch!
        yaw = pitch
        current_pos = np.array([-st.position[2], -st.position[0], st.position[1]]) # 变成 正常世界坐标系, x前 y左 z上

        for i in range(control_steps):
            traj_length = np.linalg.norm(traj_cam[0][-1][:2])
            follower.control(current_pos, yaw, np.array(trajectory_points_world, dtype=np.float32), look_ahead_dist, traj_length)
            steps += 1

            #print("success:", env.get_metrics()['success'], "distance:", env.get_metrics()['distance_to_goal'], "traj_length:", traj_length)
            # 成功条件
            cur_p = np.array(env.sim.get_agent_state(0).position, dtype=float)
            if env.get_metrics()['distance_to_goal'] <= success_distance and traj_length <= 1.5:
                success = True

            traj_walk.append([cur_p[0], cur_p[1], cur_p[2]])
            last_obs = env.sim.get_sensor_observations()

            # 录像（可选）
            if vw is not None:
                path_graph_vis = visualize_path_on_graph(
                    env,
                    graph = robot.graph,
                    node_path=path_nodes,
                    show_nodes= True,
                    show_edges= True,
                    path_thickness=2,
                    node_radius=10,
                    canvas_h=1024,
                    predicted_traj_world=trajectory_points_world,
                    executed_world_path=traj_walk,
                )

                rgb_im = _ensure_uint8_rgb3(last_obs["rgb"])

                # depth_im = last_obs["depth"]
                # if depth_im.ndim == 3:
                #     depth_im = depth_im[..., 0]  # 去掉多余通道
                # depth_im = np.nan_to_num(depth_im)  # 去掉 NaN
                # depth_im = np.clip(depth_im / depth_im.max(), 0, 1)  # 归一化到 0-1
                # depth_im = (depth_im * 255).astype(np.uint8)
                # depth_im = np.stack([depth_im] * 3, axis=-1)  # 变成 RGB3

                frame = make_vw_frame(rgb_im, gola_im_vis, path_graph_vis, panel_h=900, step=steps, 
                                        traj_len=np.linalg.norm(traj_cam[0][-1][:2]), 
                                        goal_distance=env.get_metrics()['distance_to_goal'],
                                        imagegoal_mode = True)
                vw.append_data(frame)

            if success:
                break

        if success:
            break

    if vw is not None:
        vw.close()

    # 计算实际路径长度
    actual_path_length = 0.0
    for i in range(len(traj_walk) - 1):
        p0 = np.asarray(traj_walk[i],   dtype=float)
        p1 = np.asarray(traj_walk[i+1], dtype=float)
        actual_path_length += np.linalg.norm(p1[[0, 2]] - p0[[0, 2]])

    return {"success": int(success), "steps": int(steps), "path_length": float(actual_path_length)}, predict_trj, stuck_counter

# ========== 添加场景列表 ==========
SCENE_ID_MAP = {
    "00877-4ok3usBNeis": "4ok3usBNeis", "00853-5cdEh9F2hJL": "5cdEh9F2hJL",
    "00890-6s7QHgap2fW": "6s7QHgap2fW", 
    "00823-7MXmsvcQjpJ": "7MXmsvcQjpJ",
    "00849-a8BtkwhxdRV": "a8BtkwhxdRV", 
    "00827-BAbdmeyTvMZ": "BAbdmeyTvMZ",
    "00873-bxsVRursffK": "bxsVRursffK",
    "00810-CrMo8WxCyVb": "CrMo8WxCyVb", "00891-cvZr5TUy5C5": "cvZr5TUy5C5",
    "00824-Dd4bFSTQ8gi": "Dd4bFSTQ8gi", "00843-DYehNKdT76V": "DYehNKdT76V",
    "00821-eF36g7L6Z9M": "eF36g7L6Z9M", "00861-GLAQ4DNUx5U": "GLAQ4DNUx5U",
    "00815-h1zeeAwLh9Z": "h1zeeAwLh9Z", "00894-HY1NcmCgn3n": "HY1NcmCgn3n",
    "00862-LT9Jq6dN3Ea": "LT9Jq6dN3Ea",
    "00869-MHPLjHsuG27": "MHPLjHsuG27", 
    "00820-mL8ThkuaVTM": "mL8ThkuaVTM",
    "00876-mv2HUxq3B53": "mv2HUxq3B53", 
    "00880-Nfvxx8J5NCo": "Nfvxx8J5NCo",
    "00800-TEEsavR23oF": "TEEsavR23oF",
    "00814-p53SfW6mjZe": "p53SfW6mjZe", "00835-q3zU7Yy5E5s": "q3zU7Yy5E5s",
    "00829-QaLdnwvtxbs": "QaLdnwvtxbs",
    "00832-qyAac8rV8Zk": "qyAac8rV8Zk", "00813-svBbv1Pavdk": "svBbv1Pavdk",
    "00871-VBzV5z6i1WS": "VBzV5z6i1WS",
    "00878-XB4GS9ShBRE": "XB4GS9ShBRE",
    "00831-yr17PDCnDDW": "yr17PDCnDDW",
    "00848-ziup5kvtCCR": "ziup5kvtCCR", 
    "00839-zt1RVoi7PcG": "zt1RVoi7PcG",
    "00802-wcojb4TFT35": "wcojb4TFT35", 
    "00808-y9hTuugGdiq": "y9hTuugGdiq",
    "00803-k1cupFYWXJ6": "k1cupFYWXJ6",
    "00844-q5QZSEeHe5g": "q5QZSEeHe5g",
    "00847-bCPU9suPUw9": "bCPU9suPUw9",
}


# question scene: /Users/wangbo/Desktop/PhD文章阅读/VLA/结果展示/all_scene_episodes.csv
# SCENE_ID_MAP = {
#     # "带前缀的完整ID": "不带前缀的短ID",
#     # "00820-mL8ThkuaVTM": "mL8ThkuaVTM",
#     # "00821-eF36g7L6Z9M": "eF36g7L6Z9M",
#     # "00823-7MXmsvcQjpJ": "7MXmsvcQjpJ",
#     # "00831-yr17PDCnDDW": "yr17PDCnDDW",
#     # "00847-bCPU9suPUw9": "bCPU9suPUw9", 
#     # "00862-LT9Jq6dN3Ea": "LT9Jq6dN3Ea",
#     # "00878-XB4GS9ShBRE": "XB4GS9ShBRE",
#     # "00839-zt1RVoi7PcG": "zt1RVoi7PcG",
#     # "00844-q5QZSEeHe5g": "q5QZSEeHe5g",
#     "00802-wcojb4TFT35": "wcojb4TFT35", 
#     "00808-y9hTuugGdiq": "y9hTuugGdiq",
#     "00803-k1cupFYWXJ6": "k1cupFYWXJ6",
# }

HABITAT_ROOT_DIR = "third-party/habitat-lab"
# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser("Grparserh+NavDP follow")
    # —— env / dataset ——（直接复用你 run_imagenav_localize.py 的参数）
    parser.add_argument("--benchmark_dataset", type=str, default="hm3d", choices=["hm3d", "mp3d"])
    parser.add_argument("--HM3D_CONFIG_PATH", type=str, default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml")
    parser.add_argument("--HM3D_EPISODE_PREFIX", type=str, default="data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz")
    parser.add_argument("--HM3D_SCENE_PREFIX", type=str, default="/nas_dataset/wangbo/HM3D")
    parser.add_argument("--content_scenes", type=str, default="4ok3usBNeis", help="choose the specific scene")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--pitch_deg", type=float, default=-10.0, help="set the bird's eye view")
    parser.add_argument("--image_hfov", type=float, default=90)
    parser.add_argument("--sensor_height", type=float, default=1.0)
    parser.add_argument("--nav_task", type=str, default='imgnav')

    # actions
    parser.add_argument("--move_forward", type=float, default=0.25)
    parser.add_argument("--turn_left", type=int, default=30)
    parser.add_argument("--turn_right", type=int, default=30)

    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=1000)
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
    parser.add_argument("--rpc_port", type=int, default=7777)

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
    preload_dinov2 = torch.hub.load('/home/wangbo/codes/BSC-Nav/third-party/dinov2', args.dino_size, source='local').to('cuda')

    # prepare groundedsam2
    preload_gsam2 = GroundedSAM2(
        sam2_checkpoint = "/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        gdino_id = "/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2/grounding-dino-tiny",
        default_box_threshold = 0.25,
        default_text_threshold = 0.25,
        device="cuda",
    )

    # === 🆕 场景遍历：对 SCENE_ID_MAP 中的每个场景独立跑一遍 ===
    MEMORY_ROOT = "/nas_home/wangbo/vis_nav/memory"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/all_finetuned_navdp_goal_match_baseline_m1.0_r0.5/question_scene"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/trainednavdp_goal_match_baseline_m1.0_r0.5/question_scene"
    # BASE_OUT = "/nas_home/wangbo/vis_nav/result_hm3d/goal_match_baseline_m1.0_r0.5/question_scene"
    BASE_OUT = f"/nas_home/wangbo/vis_nav/result_hm3d/trainednavdp_goal_match_baseline_m{args.min_dis}_r{args.radius}"
    all_scenes_results = []  # 汇总每个 scene 的简报

    skipped_scenes = []
    single_floor_scenes = []
    scene_items = list(SCENE_ID_MAP.items())

    for scene_full_id, scene_short_id in tqdm(scene_items, desc="Processing scenes", ncols=100):
        print(f"\n========== [SCENE] {scene_full_id} (short={scene_short_id}) ==========")

        # === 🆕 动态切换本场景的输入/输出路径 & content_scenes ===
        args.content_scenes = scene_short_id
        scene_mem_dir = os.path.join(MEMORY_ROOT, scene_full_id)
        args.graph_json  = os.path.join(scene_mem_dir, f"place_graph_min{args.min_dis}_radius{args.radius}_floor{args.floor_idx}.json")
        args.explore_npz = os.path.join(scene_mem_dir, "explore_log.npz")
        args.floor_json  = os.path.join(scene_mem_dir, "floor_data.json")
        # 为该 scene 单独的输出目录
        args.out_dir = os.path.join(BASE_OUT, f"floor_{args.floor_idx}", scene_full_id)
        os.makedirs(args.out_dir, exist_ok=True)

        skip_csv_path = os.path.join(BASE_OUT, f"floor_{args.floor_idx}", "skip_scene_no_floor.csv")
        single_floor_csv_path = os.path.join(BASE_OUT, f"floor_{args.floor_idx}", "single_floor_scenes.csv")

        floor_filtering = True
        # 检查floor_json是否存在
        if not os.path.exists(args.floor_json):
            # ***** 打印警告信息时也使用 long_id *****
            print(f"[WARN] 场景 {scene_full_id} 的 floor_json 不存在: {args.floor_json}")
            print(f"跳过场景 {scene_full_id}")

            skipped_scenes.append({
            'scene_id': scene_full_id,
            'reason': 'floor_json_not_found',
            'file_checked': args.floor_json
            })
            continue

        floor_range, num_floors = set_floor_filter_from_json(
            floor_json_path=args.floor_json, 
            floor_idx=args.floor_idx
        )

        if floor_range is False:
            print(f"{args.floor_idx} is out of range, this scene has just {num_floors} layes")
            continue
        
        # 打印楼层信息
        if num_floors <= 1:
            print(f"[场景{scene_full_id}] 单层场景，不进行楼层过滤")

            single_floor_scenes.append({
            'scene_id': scene_full_id,
            'num_floors': num_floors,
            'note': 'Single floor scene, no filtering applied'
            })

            # ★ 新增逻辑：如果是单层且我们要跑 floor_idx==1，则跳过
            if int(args.floor_idx) >= 1:
                print(f"[场景{scene_full_id}] 单层 + floor_idx={args.floor_idx} → 跳过此场景")
                skipped_scenes.append({
                    'scene_id': scene_full_id,
                    'reason': f'single floor, skip due to floor{args.floor_idx}',
                    'file_checked': args.floor_json
                })
                # 这里 env 已经创建了，记得关闭
                try:
                    env.close()
                except Exception:
                    pass
                continue  # ← 直接跳过本场景

            # 否则继续正常流程，只是关闭楼层过滤
            floor_filtering = False

        else:
            print(f"[场景{scene_full_id}] 多层场景（共{num_floors}层），"
                    f"当前评估第{args.floor_idx}层，过滤范围: {floor_range}")

        # 1) 机器人 & 环境
        robot = ImageNavGraphRobot(args,
                                graph_json=args.graph_json,
                                explore_npz=args.explore_npz,
                                preload_gsam=preload_gsam2, 
                                preload_dino=preload_dinov2)
        env = get_objnav_env(args)



        # 判断数据里面有多少个episodes，避免episode设置过大而出错
        try:
            total_eps = len(env._dataset.episodes)
        except Exception:
            total_eps = args.eval_episodes  

        num_to_run = min(args.eval_episodes, total_eps)
        print("eval episode:", num_to_run)

        controller = HabitatController(args, env)
        control_steps = int(args.control_time * args.control_freq)
        print(f"control fre: {args.control_freq}Hz, control_time: {args.control_time}s")
        print(f"agent will execute {control_steps} steps for each trajectory!")

        results = []
        result_id = 0
        episode_save = []

        stuck_episode = []

        for ep in range(num_to_run):
            obs_init = env.reset()

            if floor_filtering:
                # ── 楼层过滤：先拿“真”起终点（世界坐标），不合楼层就跳过 ──
                try:
                    s_xyz, g_xyz = robot.get_true_start_goal_positions(env)  # 你已有此函数
                except Exception as e:
                    print(f"[Episode {ep}] get_true_start_goal_positions() failed: {e}")
                    continue
                
                # 在楼层筛选这里做两个改动：
                # 如果是最高一层，不做最高高度过滤
                # 如果是最低一层，不做最低高度过滤
                s_ok = robot.on_graph_floor(s_xyz[1])
                g_ok = (g_xyz is None) or robot.on_graph_floor(g_xyz[1])  # 有的 image-goal 可能没有真坐标
                if not (s_ok and g_ok):
                    print(f"[Episode {ep}] skip (out of floor): "
                        f"start_y={s_xyz[1]:.3f}, goal_y={'None' if g_xyz is None else f'{g_xyz[1]:.3f}'}, "
                        f"allowed={robot.floor_filter}")
                    continue
                
                episode_save.append(ep)

            # ---------- 保存起点/目标图像（JPG） ----------
            image_dir = os.path.join(args.out_dir, "start_goal_image")
            os.makedirs(image_dir, exist_ok=True)
            start_img = _ensure_uint8_rgb3(obs_init["rgb"])
            Image.fromarray(start_img).save(os.path.join(image_dir, f"ep_{result_id:04d}_start.jpg"), quality=95)

            goal_img = obs_init["instance_imagegoal"]  # 你在 ImageNavGraphRobot 里已有这个函数
            goal_img = _ensure_uint8_rgb3(goal_img)
            Image.fromarray(goal_img).save(os.path.join(image_dir, f"ep_{result_id:04d}_goal.jpg"), quality=95)

            # ----------visualize the constructed graph on the specific scene ----------
            # graph_only_path = os.path.join(args.out_dir, f"graph_only.png")
            # if not os.path.exists(graph_only_path):
            #     robot.visualize_graph_only(env, out_path=graph_only_path,
            #                         show_nodes=True, show_edges=True)

            #  ----------创建保存video的文件夹 ----------
            vpath = os.path.join(args.out_dir, "videos", f"ep_{result_id:03d}.mp4") if args.record_video else None
            os.makedirs(os.path.join(args.out_dir, "videos"), exist_ok=True)

            #  ----------计算最短路径，计算spl ----------
            try:
                episode = env.current_episode
                goal_pos = np.array(episode.goals[0].view_points[episode.goal_image_id].agent_state.position)
                start_pos = np.array(episode.start_position)
                shortest_path_length = env.sim.geodesic_distance(start_pos, goal_pos)
                if shortest_path_length == float('inf') or shortest_path_length < 0:
                    shortest_path_length = -1.0
            except Exception as e:
                print(f"[WARN] Failed to get geodesic distance: {e}")
                shortest_path_length = -1.0

            #  ----------推理并行走 ----------
            ret, pred_traj, stuck_num = run_episode_with_graph(
                obs_init,
                env, robot, follower=controller,
                rpc_host=args.rpc_host, rpc_port=args.rpc_port,
                hfov_deg=args.image_hfov, success_distance=args.success_distance,
                re_localize_every=args.re_localize_every,
                max_total_steps=args.max_total_steps,
                look_ahead_dist=args.look_ahead_dist,
                control_steps = control_steps,
                video_path=vpath,
                stuck_check_steps=args.stuck_check_steps,
                stuck_threshold=args.stuck_threshold,
            )
            print("controller reset")
            controller.reset()

            # 计算 SPL
            if ret['success'] and shortest_path_length > 0:
                spl = shortest_path_length / max(ret['path_length'], shortest_path_length)
            else:
                spl = 0.0

            # 保存结果
            result_dict = {
                'success': ret['success'],
                'steps': ret['steps'],
                'spl': spl
            }
            results.append(result_dict)

            result_id += 1

            # 保存stuck超过最大次数的episode
            if stuck_num >=5:
                stuck_episode.append(result_id)

            # 计算当前累计的 SR 和 SPL
            current_sr = sum(r['success'] for r in results) / len(results)
            current_spl = sum(r['spl'] for r in results) / len(results)
            
            print(f"[EP{result_id}] success={ret['success']}, steps={ret['steps']}, "
                  f"SR={current_sr:.4f}, SPL={current_spl:.4f}")


        # ========== 单个场景的结果汇总 ==========
        if results:
            scene_sr = sum(r['success'] for r in results) / len(results)
            scene_spl = sum(r['spl'] for r in results) / len(results)
        else:
            scene_sr = 0.0
            scene_spl = 0.0

        # 写个最简 CSV
        csv_path = os.path.join(args.out_dir, "metric.csv")
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "success", "steps", "SR", "SPL"])

            for i, r in enumerate(results):
                w.writerow([i, r["success"], r["steps"], "", f"{r['spl']:.4f}"])
            w.writerow([])
            w.writerow(["OVERALL", "", "", f"{scene_sr:.4f}", f"{scene_spl:.4f}"])
        
        print(f"[场景 {scene_full_id}] SR={scene_sr:.4f}, SPL={scene_spl:.4f}")
        print(f"[DONE] saved {csv_path}")

        # 写一个episode总结
        file_name = "episode_summary.txt"
        output_path = os.path.join(args.out_dir, file_name)
        floor_info = f"单层场景（不过滤）" if num_floors <= 1 else f"多层场景（共{num_floors}层），评估第{args.floor_idx}层"

        content_to_write = f"""Scene: {args.content_scenes}
        Total Episode: {num_to_run}
        Floor: {args.floor_idx}
        current floor episode: {result_id}
        filtered episode: {episode_save}
        SR: {scene_sr:.4f}
        SPL: {scene_spl:.4f}
        episode_stuck: {stuck_episode}
        """

        # 4. 写入文件
        try:
            with open(output_path, 'w') as f:
                f.write(content_to_write)
            print(f"✅ 实验episode文件已保存至: {output_path}")

            # 添加到总体结果
            all_scenes_results.append({
                'scene': scene_full_id,
                'episodes': result_id,
                'sr': scene_sr,
                'spl': scene_spl
            })
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

        if skipped_scenes:
            print(f"\n[INFO] 共跳过 {len(skipped_scenes)} 个场景。正在写入记录到 {skip_csv_path}")
            
            # 确定 CSV 文件的列名（表头）
            fieldnames = ['scene_id', 'reason', 'file_checked']
            
            try:
                # 使用 'w' 模式写入文件，newline='' 防止空行
                with open(skip_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    # 创建 DictWriter 对象，它可以处理字典列表
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)      
                    # 写入表头
                    writer.writeheader()    
                    # 写入所有记录
                    writer.writerows(skipped_scenes) 
                print("[INFO] CSV 文件写入成功。")
            except Exception as e:
                print(f"[ERROR] 写入 CSV 文件失败: {e}")

        if single_floor_scenes:
            print(f"\n[INFO] 共识别 {len(single_floor_scenes)} 个单层场景。正在写入记录到 {single_floor_csv_path}")
            # 确定 CSV 文件的列名（表头）
            fieldnames = ['scene_id', 'num_floors', 'note']
            try:
                # 使用 'w' 模式写入文件
                with open(single_floor_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(single_floor_scenes)  
                print("[INFO] 单层场景 CSV 文件写入成功。")
            except IOError as e:
                print(f"[ERROR] 写入单层场景 CSV 文件失败: {e}")

        # 清理环境
        env.close()

    # ========== 所有场景的总体汇总 ==========
    if all_scenes_results:
        overall_csv_path = os.path.join(BASE_OUT, f"floor_{args.floor_idx}", "overall_results.csv")
        with open(overall_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Scene", "Episodes", "SR", "SPL"])
            
            total_episodes = 0
            weighted_sr = 0.0
            weighted_spl = 0.0
            
            for scene_result in all_scenes_results:
                w.writerow([
                    scene_result['scene'],
                    scene_result['episodes'],
                    f"{scene_result['sr']:.4f}",
                    f"{scene_result['spl']:.4f}"
                ])
                
                # 加权平均（按episode数量）
                total_episodes += scene_result['episodes']
                weighted_sr += scene_result['sr'] * scene_result['episodes']
                weighted_spl += scene_result['spl'] * scene_result['episodes']
            
            # 计算加权平均
            if total_episodes > 0:
                avg_sr = weighted_sr / total_episodes
                avg_spl = weighted_spl / total_episodes
            else:
                avg_sr = 0.0
                avg_spl = 0.0
            
            w.writerow([])
            w.writerow(["AVERAGE", total_episodes, f"{avg_sr:.4f}", f"{avg_spl:.4f}"])
        
        print(f"\n{'='*60}")
        print(f"所有场景评估完成！")
        print(f"总体平均 SR: {avg_sr:.4f}, SPL: {avg_spl:.4f}")
        print(f"详细结果保存在: {overall_csv_path}")
        print(f"{'='*60}")
if __name__ == "__main__":
    main()
