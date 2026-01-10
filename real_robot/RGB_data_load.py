from pathlib import Path
import numpy as np
import cv2
from rosbags.highlevel import AnyReader

root_dir = '/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52'
bag_path = Path(root_dir)  # rosbag2 文件夹

RGB_TOPIC = "/camera/camera/color/image_raw"

OUT_DIR = Path(root_dir) / "rgbd_export"
RGB_DIR = OUT_DIR / "rgb"
RGB_DIR.mkdir(parents=True, exist_ok=True)

# 写 PNG 更快：压缩等级设低一点（0最快、文件最大；3折中；默认3）
PNG_PARAMS = [cv2.IMWRITE_PNG_COMPRESSION, 1]

def imgmsg_to_numpy_rgb(msg):
    """
    只处理 RGB topic 常见编码：rgb8 / bgr8 / mono8
    返回 BGR (H,W,3) 或 mono (H,W)
    """
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    step = msg.step

    if enc in ("rgb8", "bgr8"):
        # 注意：step 是 bytes per row，可能大于 w*3（有 padding）
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = buf.reshape(h, step)[:, :w * 3].reshape(h, w, 3)

        if enc == "rgb8":
            # 转 BGR 以便 cv2.imwrite 正常保存
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if enc == "mono8":
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = buf.reshape(h, step)[:, :w].reshape(h, w)
        return img

    raise ValueError(f"Unsupported RGB encoding: {msg.encoding}")

with AnyReader([bag_path]) as reader:
    rgb_conns = [c for c in reader.connections if c.topic == RGB_TOPIC]
    if not rgb_conns:
        raise RuntimeError(f"RGB topic not found: {RGB_TOPIC}")

    saved = 0
    for c, t_ns, raw in reader.messages(connections=rgb_conns):
        msg = reader.deserialize(raw, c.msgtype)
        rgb = imgmsg_to_numpy_rgb(msg)

        stem = f"{t_ns}"
        out_file = RGB_DIR / f"{stem}.png"
        ok = cv2.imwrite(str(out_file), rgb, PNG_PARAMS)
        if not ok:
            raise RuntimeError(f"Failed to write image: {out_file}")

        saved += 1
        if saved % 200 == 0:
            print(f"Saved {saved} RGB frames...")

print(f"Done. Saved RGB frames: {saved}")
print(f"Output folder: {OUT_DIR.resolve()}")
