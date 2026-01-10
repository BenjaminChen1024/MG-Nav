from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import Stores, get_typestore

from pathlib import Path
import os
import numpy as np
import cv2
from rosbags.highlevel import AnyReader

root_dir = '/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52'
bag_path = Path(root_dir)  # 注意：是文件夹

RGB_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/aligned_depth_to_color/image_raw"

OUT_DIR = Path(root_dir + "/rgbd_export")
RGB_DIR = OUT_DIR / "rgb"
DEPTH_DIR = OUT_DIR / "depth"

RGB_DIR.mkdir(parents=True, exist_ok=True)
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

def imgmsg_to_numpy(msg):
    """支持常见的 sensor_msgs/Image：rgb8/bgr8/mono8/16UC1/32FC1"""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()

    # 原始buffer
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        # step 可能包含行对齐padding，这里按 step reshape 再裁剪
        row = msg.step
        img = buf.reshape(h, row)[:, :w*3].reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 统一保存BGR给opencv
        return img

    if enc == "mono8":
        row = msg.step
        img = buf.reshape(h, row)[:, :w].reshape(h, w)
        return img

    if enc == "16uc1":
        # 16位深度：data 实际是 uint16
        buf16 = np.frombuffer(msg.data, dtype=np.uint16)
        # step 是字节数，所以每行像素数 = step/2
        row_pix = msg.step // 2
        img = buf16.reshape(h, row_pix)[:, :w].reshape(h, w)
        return img

    if enc == "32fc1":
        # 32位float深度：data 实际是 float32
        buf32 = np.frombuffer(msg.data, dtype=np.float32)
        row_pix = msg.step // 4
        img = buf32.reshape(h, row_pix)[:, :w].reshape(h, w)
        return img

    raise ValueError(f"Unsupported encoding: {msg.encoding}")

def save_depth(depth, path: Path):
    """优先保真保存：16UC1 -> 16-bit PNG；32FC1 -> 32-bit TIFF"""
    if depth.dtype == np.uint16:
        cv2.imwrite(str(path.with_suffix(".png")), depth)  # 16-bit png
    elif depth.dtype == np.float32:
        # 32F 用 tiff 更稳（png不支持32F）
        cv2.imwrite(str(path.with_suffix(".tiff")), depth)
    else:
        # 兜底：转16位
        d = np.clip(depth, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(path.with_suffix(".png")), d)

with AnyReader([bag_path]) as reader:
    conns = reader.connections
    rgb_conns = [c for c in conns if c.topic == RGB_TOPIC]
    dep_conns = [c for c in conns if c.topic == DEPTH_TOPIC]

    if not rgb_conns:
        raise RuntimeError(f"RGB topic not found: {RGB_TOPIC}")
    if not dep_conns:
        raise RuntimeError(f"Depth topic not found: {DEPTH_TOPIC}")

    # 先把 depth 全读出来，存 (t_ns, depth_array)
    depth_times = []
    depth_imgs = []
    for c, t_ns, raw in reader.messages(connections=dep_conns):
        msg = reader.deserialize(raw, c.msgtype)
        depth = imgmsg_to_numpy(msg)
        depth_times.append(t_ns)
        depth_imgs.append(depth)

    depth_times = np.array(depth_times, dtype=np.int64)

    print(f"Loaded depth frames: {len(depth_times)}")

    # 遍历 RGB，每帧找最近的 depth（按bag记录时间 t_ns）
    saved = 0
    for c, t_ns, raw in reader.messages(connections=rgb_conns):
        msg = reader.deserialize(raw, c.msgtype)
        rgb = imgmsg_to_numpy(msg)

        # 最近邻对齐 depth
        # idx = int(np.argmin(np.abs(depth_times - t_ns)))
        # depth = depth_imgs[idx]
        # dt_ms = abs(int(depth_times[idx] - t_ns)) / 1e6

        # 文件名用 rgb 的时间戳
        stem = f"{t_ns}"
        cv2.imwrite(str(RGB_DIR / f"{stem}.png"), rgb)
        # save_depth(depth, DEPTH_DIR / stem)

        saved += 1
        if saved % 50 == 0:
            print(f"Saved {saved} RGBD pairs...")

print(f"Done. Saved RGBD pairs: {saved}")
print(f"Output folder: {OUT_DIR.resolve()}")