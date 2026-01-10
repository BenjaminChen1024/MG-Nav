import csv
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader

root_dir = '/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/'
BAG_DIR   = Path(root_dir)
RGB_TOPIC = "/camera/camera/color/image_raw"
ODOM_TOPIC = "/odom"

out_csv = root_dir + "/rgbd_export/rgb_xyz_quat.csv"

rgb_times = []    # [t_ns]
odom_times = []   # [t_ns]
odom_pose = []    # [(x,y,z,qx,qy,qz,qw)]

with AnyReader([BAG_DIR]) as reader:
    rgb_conns  = [c for c in reader.connections if c.topic == RGB_TOPIC]
    odom_conns = [c for c in reader.connections if c.topic == ODOM_TOPIC]

    if not rgb_conns:
        raise RuntimeError(f"Missing RGB topic: {RGB_TOPIC}")
    if not odom_conns:
        raise RuntimeError(f"Missing Odom topic: {ODOM_TOPIC}")

    # 1) 读 odom：时间戳 + pose (xyz + quat)
    for c, t_ns, raw in reader.messages(connections=odom_conns):
        msg = reader.deserialize(raw, c.msgtype)

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        odom_times.append(t_ns)
        odom_pose.append((
            float(p.x), float(p.y), float(p.z),
            float(q.x), float(q.y), float(q.z), float(q.w)
        ))

    odom_times = np.array(odom_times, dtype=np.int64)
    odom_pose = np.array(odom_pose, dtype=np.float64)  # shape [N,7]

    print("Loaded odom:", len(odom_times))

    # 2) 遍历 RGB：记录每帧时间戳
    for c, t_ns, raw in reader.messages(connections=rgb_conns):
        rgb_times.append(t_ns)

rgb_times = np.array(rgb_times, dtype=np.int64)
print("Loaded RGB:", len(rgb_times))

# 3) 最近邻匹配：每个 rgb_time 找最近的 odom_time
idx = np.searchsorted(odom_times, rgb_times, side="left")
idx = np.clip(idx, 1, len(odom_times) - 1)

left = idx - 1
right = idx

choose_right = (np.abs(odom_times[right] - rgb_times) < np.abs(odom_times[left] - rgb_times))
best = np.where(choose_right, right, left)

matched_pose = odom_pose[best]  # [M,7]
dt_ms = np.abs(odom_times[best] - rgb_times) / 1e6

# 4) 写 CSV
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "rgb_t_ns",
        "x", "y", "z",
        "qx", "qy", "qz", "qw",
        "matched_odom_t_ns", "dt_ms"
    ])

    for t, pose7, to, d in zip(rgb_times, matched_pose, odom_times[best], dt_ms):
        x, y, z, qx, qy, qz, qw = pose7
        w.writerow([int(t), x, y, z, qx, qy, qz, qw, int(to), float(d)])

print("Saved:", out_csv)
print("Median dt(ms):", float(np.median(dt_ms)))
print("95% dt(ms):", float(np.percentile(dt_ms, 95)))