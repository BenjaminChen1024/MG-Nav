import pandas as pd

# 定义文件路径
file_path = "/home/wangbo/codes/NavDP/InternData-N1-mini/vln_n1/traj_data/matterport3d_zed/17DRP5sb8fy/trajectory_90/data/chunk-000/episode_000000.parquet"

# 读取 parquet 文件
try:
    # 使用 pandas 读取文件
    data = pd.read_parquet(file_path)
    
    # 打印文件内容
    print(data)
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"读取文件时出错: {e}")