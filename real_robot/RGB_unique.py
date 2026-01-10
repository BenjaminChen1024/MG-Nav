import csv
import shutil
from pathlib import Path
from collections import OrderedDict

# ===== 路径配置 =====
root_dir = '/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/'
CSV_PATH = Path(root_dir + "/rgbd_export/rgb_xyz_quat.csv")
SRC_RGB_DIR = Path(
    root_dir + "rgbd_export/rgb"
)
DST_RGB_DIR = SRC_RGB_DIR.parent / "rgb_unique"
OUT_CSV = CSV_PATH.parent / "rgb_xyz_quat_unique.csv"

DST_RGB_DIR.mkdir(parents=True, exist_ok=True)

# ===== 读取 CSV =====
rows = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

# ===== (x,y) 去重，只保留第一次出现的 =====
seen = OrderedDict()   # (x,y) -> row
duplicates = []

for r in rows:
    key = (r["x"], r["y"])  # 严格相等
    if key not in seen:
        seen[key] = r
    else:
        duplicates.append(r)

print("Total frames:", len(rows))
print("Unique (x,y):", len(seen))
print("Filtered out:", len(duplicates))

# ===== copy RGB 文件 =====
copied = 0
missing = 0

for r in seen.values():
    src = SRC_RGB_DIR / f'{r["rgb_t_ns"]}.png'
    dst = DST_RGB_DIR / f'{r["rgb_t_ns"]}.png'
    if src.exists():
        shutil.copy2(src, dst)  # copy2 保留时间戳等信息
        copied += 1
    else:
        print("[WARN] Missing:", src)
        missing += 1

# ===== 写新的 CSV =====
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in seen.values():
        writer.writerow(r)

print("\nDone.")
print("Copied images:", copied)
print("Missing images:", missing)
print("New RGB folder:", DST_RGB_DIR.resolve())
print("New CSV:", OUT_CSV.resolve())
