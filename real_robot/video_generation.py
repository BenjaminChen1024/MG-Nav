import cv2
from pathlib import Path

img_dir = Path("/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_05-15_02_23/rgbd_export/rgb_unique")
out_path = "rgb_unique.mp4"
fps = 30

imgs = sorted(img_dir.glob("*.png"))
assert len(imgs) > 0, "No PNG images found"

# 读第一张确定尺寸
first = cv2.imread(str(imgs[0]))
h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for p in imgs:
    img = cv2.imread(str(p))
    if img is None:
        print("Skip:", p)
        continue
    writer.write(img)

writer.release()
print("Saved:", out_path)
