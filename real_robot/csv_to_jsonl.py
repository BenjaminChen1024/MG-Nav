import csv
import json
import math
from pathlib import Path
import numpy as np

# ====== 你需要改的路径 ======
CSV_PATH = Path("/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/rgbd_export/rgb_xyz_quat_unique.csv")
RGB_DIR  = Path("/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/rgbd_export/rgb_unique")
DEPTH_DIR = Path("/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/rgbd_export/depth")
OUT_JSONL = CSV_PATH.parent / "obs/frames_meta.jsonl"

EPISODE_ID = 1
SUBGOAL_ID = 0
ACTION = "na"   # 你说随便填一个字符串即可

# 位置：ROS -> Habitat 的基变换矩阵（v_h = M v_ros）
# x_h = -y_ros, y_h = z_ros, z_h = -x_ros
M = np.array([
    [ 0.0, -1.0,  0.0],
    [ 0.0,  0.0,  1.0],
    [-1.0,  0.0,  0.0],
], dtype=np.float64)

def quat_xyzw_to_R(qx, qy, qz, qw):
    """ROS quat (x,y,z,w) -> rotation matrix (3x3)."""
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)],
        [  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)],
    ], dtype=np.float64)

def R_to_quat_wxyz(R):
    """rotation matrix -> quat (w,x,y,z)."""
    # 稳健实现
    t = np.trace(R)
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / S
            x = 0.25 * S
            y = (R[0,1] + R[1,0]) / S
            z = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / S
            x = (R[0,1] + R[1,0]) / S
            y = 0.25 * S
            z = (R[1,2] + R[2,1]) / S
        else:
            S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / S
            x = (R[0,2] + R[2,0]) / S
            y = (R[1,2] + R[2,1]) / S
            z = 0.25 * S

    # 归一化
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    return [w/norm, x/norm, y/norm, z/norm]

def ros_pose_to_habitat(xr, yr, zr, qx, qy, qz, qw):
    """ROS pose -> Habitat pose. 返回 (xh,yh,zh, quat_wxyz, yaw)."""
    # position
    v_ros = np.array([xr, yr, zr], dtype=np.float64)
    v_h = M @ v_ros
    xh, yh, zh = float(v_h[0]), float(v_h[1]), float(v_h[2])

    # orientation: R_h = M R_ros M^T（正交基变换）
    R_ros = quat_xyzw_to_R(qx, qy, qz, qw)
    R_h = M @ R_ros @ M.T
    quat_wxyz = R_to_quat_wxyz(R_h)

    # yaw：用“forward = -z”定义，取 body forward 在世界坐标的投影角
    # body forward 向量（Habitat agent）默认为 local [0,0,-1]
    f = R_h @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    yaw = float(math.atan2(f[0], -f[2]))  # 0时朝向 -z

    return xh, yh, zh, quat_wxyz, yaw

def find_depth_path(stem: str):
    """兼容你的 depth 可能是 .npy/.png/.tiff 的情况，优先 .npy。"""
    for ext in [".npy", ".png", ".tiff", ".tif"]:
        p = DEPTH_DIR / f"{stem}{ext}"
        if p.exists():
            return str(p)
    # 找不到也返回一个默认npy路径，方便你后续再补齐
    return str(DEPTH_DIR / f"{stem}.npy")

# ====== 生成 jsonl ======
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

with open(CSV_PATH, "r") as f_in, open(OUT_JSONL, "w") as f_out:
    reader = csv.DictReader(f_in)

    for frame_id, r in enumerate(reader):
        t_ns = r["rgb_t_ns"]
        xr, yr, zr = float(r["x"]), float(r["y"]), float(r["z"])
        qx, qy, qz, qw = float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])

        xh, yh, zh, quat_wxyz, yaw = ros_pose_to_habitat(xr, yr, zr, qx, qy, qz, qw)

        rgb_path = str(RGB_DIR / f"{t_ns}.png")
        depth_path = find_depth_path(t_ns)

        item = {
            "frame_id": frame_id,
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "pose": {
                "x": xh,
                "y": yh,
                "z": zh,
                "yaw": yaw,
                "quat": quat_wxyz,   # Habitat: [w, x, y, z]
            },
            "episode_id": EPISODE_ID,
            "subgoal_id": SUBGOAL_ID,
            "traj_step": frame_id,
            "action": ACTION,
        }

        f_out.write(json.dumps(item) + "\n")

print("Saved jsonl:", OUT_JSONL)
