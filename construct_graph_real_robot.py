
from scipy.__config__ import show
from scipy._lib.array_api_compat.numpy import False_
from wangbo_place_graph_builder_obs import PlaceGraphBuilder, SemanticSaver, to_np_rgb3, imread_rgb_uint8, normalize_sem_to_object_list, build_frame_object_features_once
from env import NavEnv, get_objnav_env 
from wangbo_occupancy_map import draw_explored_on_habitat_topdown
import torch 
import json
import os, math
import numpy as np
import cv2
import sys
import argparse
from tqdm import tqdm
import pickle
sys.path.append("/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2")

from grounded_sam2_wrapper import GroundedSAM2

# ----------utils-----------
def save_feats_by_frame_pickle(path, feats_by_frame):
    # 协议 5 对大数组更友好；二进制写
    with open(path, "wb") as f:
        pickle.dump(feats_by_frame, f, protocol=5)

def load_feats_by_frame_pickle(path):
    with open(path, "rb") as f:
        feats_by_frame = pickle.load(f)
    return feats_by_frame

def load_frames_from_json(json_path):
    """
    读取逐行JSON文件，返回包含 (frame_id, rgb_path, rgb_image) 的列表。
    """
    frames = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            frame_id = data.get("frame_id")
            rgb_path = data.get("rgb_path")

            if not os.path.exists(rgb_path):
                print(f"[警告] 找不到文件: {rgb_path}")
                continue

            # 读取RGB图像（BGR格式，可根据需要转换）
            rgb_image = cv2.imread(rgb_path)
            if rgb_image is None:
                print(f"[警告] 无法读取图像: {rgb_path}")
                continue

            frames.append({
                "frame_id": frame_id,
                "rgb_path": rgb_path,
                "rgb_image": rgb_image
            })
    return frames


SCENE_ID_MAP = {
    # "带前缀的完整ID": "不带前缀的短ID",
    "00877-4ok3usBNeis": "4ok3usBNeis", "00853-5cdEh9F2hJL": "5cdEh9F2hJL",
    "00890-6s7QHgap2fW": "6s7QHgap2fW", "00823-7MXmsvcQjpJ": "7MXmsvcQjpJ",
    "00849-a8BtkwhxdRV": "a8BtkwhxdRV", "00827-BAbdmeyTvMZ": "BAbdmeyTvMZ",
    "00847-bCPU9suPUw9": "bCPU9suPUw9", "00873-bxsVRursffK": "bxsVRursffK",
    "00810-CrMo8WxCyVb": "CrMo8WxCyVb", "00891-cvZr5TUy5C5": "cvZr5TUy5C5",
    "00824-Dd4bFSTQ8gi": "Dd4bFSTQ8gi", "00843-DYehNKdT76V": "DYehNKdT76V",
    "00821-eF36g7L6Z9M": "eF36g7L6Z9M", "00861-GLAQ4DNUx5U": "GLAQ4DNUx5U",
    "00815-h1zeeAwLh9Z": "h1zeeAwLh9Z", "00894-HY1NcmCgn3n": "HY1NcmCgn3n",
    "00803-k1cupFYWXJ6": "k1cupFYWXJ6", "00862-LT9Jq6dN3Ea": "LT9Jq6dN3Ea",
    "00869-MHPLjHsuG27": "MHPLjHsuG27", "00820-mL8ThkuaVTM": "mL8ThkuaVTM",
    "00876-mv2HUxq3B53": "mv2HUxq3B53", "00880-Nfvxx8J5NCo": "Nfvxx8J5NCo",
    "00814-p53SfW6mjZe": "p53SfW6mjZe", "00835-q3zU7Yy5E5s": "q3zU7Yy5E5s",
    "00844-q5QZSEeHe5g": "q5QZSEeHe5g", "00829-QaLdnwvtxbs": "QaLdnwvtxbs",
    "00832-qyAac8rV8Zk": "qyAac8rV8Zk", "00813-svBbv1Pavdk": "svBbv1Pavdk",
    "00800-TEEsavR23oF": "TEEsavR23oF", "00871-VBzV5z6i1WS": "VBzV5z6i1WS",
    "00802-wcojb4TFT35": "wcojb4TFT35", "00878-XB4GS9ShBRE": "XB4GS9ShBRE",
    "00808-y9hTuugGdiq": "y9hTuugGdiq", "00831-yr17PDCnDDW": "yr17PDCnDDW",
    "00848-ziup5kvtCCR": "ziup5kvtCCR", "00839-zt1RVoi7PcG": "zt1RVoi7PcG",
}


MEMORY_PATH = "/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_24-21_12_52/"
SCENE_NAME = "rgbd_export"

parser = argparse.ArgumentParser("Exploration and Graph Construction")
# === 固定基础参数 ===
parser.add_argument("--benchmark_dataset", type=str, default="hm3d")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--image_hfov", type=float, default=90.0)
parser.add_argument("--sensor_height", type=float, default=1.0)
parser.add_argument("--pitch_deg", type=float, default=-10.0, help="set the bird's eye view")
parser.add_argument("--color_sensor", type=bool, default=True)
parser.add_argument("--depth_sensor", type=bool, default=True)
parser.add_argument("--semantic_sensor", type=bool, default=False)
parser.add_argument("--move_forward", type=float, default=0.25)
parser.add_argument("--turn_left", type=float, default=30.0)
parser.add_argument("--turn_right", type=float, default=30.0)
parser.add_argument("--success_distance", type=float, default=0.5)
parser.add_argument("--max_episode_steps", type=int, default=1000)

# === 数据集路径参数 ===
parser.add_argument("--dataset_dir", type=str, default="/nas_dataset/wangbo/HM3D/val/")
parser.add_argument("--scene_name", type=str, default='00877-4ok3usBNeis')
parser.add_argument("--scene_dataset_config_file", type=str,
                    default="/nas_dataset/wangbo/HM3D/hm3d_annotated_basis.scene_dataset_config.json")

# === 探索与建图参数 ===
parser.add_argument("--memory_path", type=str, default=MEMORY_PATH)
parser.add_argument("--dino_size", type=str, default="dinov2_vitl14_reg")
parser.add_argument("--random_move_num", type=int, default=30)
parser.add_argument("--floor_idx", type=int, default=1, help="if load_single_floor, single floor idx")

parser.add_argument("--min_dis", type=float, default=1.0, help='FPS min')
parser.add_argument("--radius", type=float, default=0.5, help="node radius")

parser.add_argument("--explore_map", type=bool, default=False)
parser.add_argument("--semantic_analyze", type=bool, default=False)
parser.add_argument("--construct_graph", type=bool, default=True)
parser.add_argument("--visualize_graph", type=bool, default=True)



# === 路径派生 ===
parser.add_argument("--npz_path", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "explore_log.npz"))
parser.add_argument("--out_json", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "place_graph_min1.0_radius0.5.json"))
parser.add_argument("--frames_meta_jsonl", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "frames_meta.jsonl"))
parser.add_argument("--sem_jsonl", type=str, 
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "sem/frames_sem.jsonl"))

# === Frontier Explore 参数 ===
parser.add_argument("--gs", type=int, default=1000)
parser.add_argument("--cs", type=float, default=0.1)
parser.add_argument("--depth_sample_rate", type=int, default=1000)
parser.add_argument("--min_depth", type=float, default=0.1)
parser.add_argument("--max_depth", type=float, default=10.0)
parser.add_argument("--map_height", type=float, default=10.0)
parser.add_argument("--floor_height", type=float, default=-10.0)

# === 坐标系基础参数 ===
parser.add_argument("--base_forward_axis", type=list, default=[0, 0, -1])
parser.add_argument("--base_left_axis", type=list, default=[-1, 0, 0])
parser.add_argument("--base_up_axis", type=list, default=[0, 1, 0])
parser.add_argument("--base2cam_rot", type=list, 
                    default=[1, 0, 0, 0, -1, 0, 0, 0, -1])

args = parser.parse_args()
#args.predefined_class = "seating. chest of drawers. bed. bathtub. clothes. toilet. stool. sofa. sink. tv monitor. picture. cushion. towel. shower. counter. fireplace. chair. table. gym equipment. cabinet. plant."
# args.predefined_class = ['seating', 'chest of drawers', 'bed', 'bathtub', 'clothes', 'toilet', 'stool', 'sofa', 'sink', 'tv monitor', 'picture', 'cushion', 'towel', 'shower', 'counter', 'fireplace', 'chair', 'table', 'gym equipment', 'cabinet', 'plant']
args.predefined_class = ["chair", "couch", "potted plant", "bed", "toilet", "tv", "computer", "table", "robot"]
# prepare dinov2
preload_dinov2 = torch.hub.load('/home/wangbo/codes/BSC-Nav/third-party/dinov2', args.dino_size, source='local').to('cuda')
# prepare groundedsam2
prelload_gsam2 = GroundedSAM2(
    sam2_checkpoint = "/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
    gdino_id = "/home/wangbo/codes/BSC-Nav/third-party/Grounded-SAM-2/grounding-dino-tiny",
    default_box_threshold = 0.25,
    default_text_threshold = 0.25,
    device="cuda",
)

# # 实例化
env = NavEnv(args, init_state=None, build_map=False)
builder = PlaceGraphBuilder(args, preload_dino=preload_dinov2, preload_gsam=prelload_gsam2, env=env)

# --------------探索的过程中产生的文件------------------
"""
RGBD 图像: 不变
explore_log.npz: frame_id, poses_xyz, yaws, features(每一张图的donov2 feature), quats(相机rotation xyz)
base_height: 每个观测点的y
"""


# --------------语义分割中产生的文件------------------
"""
frames_sem.jsonl: 每一个frame真的frame id, image_wh, class_names和对应的mask, input boxes等
obj_feats_by_frame.pkl: 每一个frame中的instance和对应的feature,cls feature，存成字典的形式便于后续调用
"""

if args.semantic_analyze:
    # -----------frame segmentation----------
    # semantic_save_dir = os.path.join(args.memory_path, SCENE_NAME)
    # sem_saver = SemanticSaver(semantic_save_dir)
    
    # frames_json = args.frames_meta_jsonl
    # exploration_frames = load_frames_from_json(frames_json)
    # print(f"共读取 {len(exploration_frames)} 帧。")

    # for frame in tqdm(exploration_frames):
    #     rgb_uint8 = to_np_rgb3(frame["rgb_image"])
    #     frame_id = frame["frame_id"]
    #     try:

    #         prompt = args.predefined_class
    #         det = prelload_gsam2.detect(
    #             rgb=rgb_uint8,
    #             text_prompt=prompt
    #         )
    #         H, W = rgb_uint8.shape[:2]
    #         sem_saver.write(frame_id, det, image_wh=(W, H))
    #         # --- 可选：遇到空检测也记一条轻量日志 ---
    #         if len(det["class_names"]) == 0:
    #             print(f"[INFO] GSAM no-detect at frame {frame_id}")
    #     except Exception as e:
    #         import traceback
    #         print(f"[WARN] GSAM detect failed at frame {frame_id}: {e.__class__.__name__}: {e}")
    #         traceback.print_exc(limit=1)   # 打印一行栈顶，别太吵
    # sem_saver.close()

     # -----------frame object feature extraction----------
    # 逐帧提取所有对象 crop 特征，并存成文件，便于后续的不断建图
    print('node object features extraction:')
    feats_by_frame = {}
    frames_meta = builder.load_frames_meta(args.frames_meta_jsonl)
    sem_dets_by_frame = builder.load_sem_dets(os.path.join(args.memory_path, SCENE_NAME)+ '/obs/sem/frames_sem.jsonl')
    for fid, rec in sem_dets_by_frame.items():
        meta = frames_meta.get(fid)
        if not meta or not meta.get("rgb_path"): 
            continue
        rgb_path = meta.get("rgb_path")
        if not rgb_path or not os.path.exists(rgb_path):
            continue
        rgb_uint8 = imread_rgb_uint8(rgb_path)

        class_names = rec.get("class_names", [])
        masks = rec.get("masks", [])
        obj_list = normalize_sem_to_object_list(class_names, masks)
        feat_map, cls_map = build_frame_object_features_once(builder.encoder, rgb_uint8, obj_list)
        feats_by_frame[fid] = {
            "feat": feat_map,   # {class_name: [feat_pool, ...]}
            "cls":  cls_map,    # {class_name: [feat_cls,  ...]}
        }
    print("node object features extraction done!")
    save_path = os.path.join(args.memory_path, SCENE_NAME, "obs", "sem", "obj_feats_by_frame.pkl")  # 或 .joblib
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_feats_by_frame_pickle(save_path, feats_by_frame)
# --------------建图中产生的文件------------------
"""
place_graph.json: memory graph
obj_feats: instance features [N, D], N是memory graph中存的instance feature。 place_graph中保存的有每一个instance对应的索引
"""

if args.construct_graph:

    floor_min, floor_max = None, None
    args.floor_idx = 0

    # ……之后任何时候离线构图（可多次改参数）
    print("Graph construction begin:")
    # builder.build_from_npz(
    #     npz_path=args.npz_path,
    #     out_json=args.out_json,
    #     sem_jsonl=args.sem_jsonl,
    #     cos_thresh=0.25, yaw_gate_deg=45.0,
    #     tau_len=4, knn_k=8, r_max=3.0,
    #     #仅用某一层
    #     floor_min_y=floor_min, floor_max_y=floor_max
    # )

    # node_radius_m 指范围，min_node_spacing_m是node的之间的最小间距
    load_path = os.path.join(args.memory_path, SCENE_NAME, "obs", "sem", "obj_feats_by_frame.pkl")
    obj_feats_by_frame = load_feats_by_frame_pickle(load_path)
    builder.build_spatial_node_graph(
                                npz_path=args.npz_path,
                                frames_meta_jsonl=args.frames_meta_jsonl,
                                sem_jsonl='/nas_home/wangbo/vis_nav/real_robot_experiments/data/rosbag2_2025_12_05-15_02_23/rgbd_export/obs/sem/frames_sem.jsonl',
                                node_radius_m=0.5,
                                min_node_spacing_m=1.0,
                                keyframes_per_node=4,
                                yaw_min_sep_deg=90.0,
                                y_band=0.6,
                                floor_min_y=floor_min,
                                floor_max_y=floor_max,
                                out_json=args.out_json,
                                obj_feats_by_frame=obj_feats_by_frame)
    print("Graph construction end!")

if args.visualize_graph:

    floor_range = (None, None)
    args.floor_idx = 0
    """
    可视化构建的graph
    """
    # 假设 builder 是 PlaceGraphBuilder 实例，env 是已初始化的 Habitat NavEnv
    json_path = args.out_json

    builder.visualize_graph_from_json_with_keyframes_real_robot(
        env, 
        graph_json_path=json_path, 
        out_path=args.memory_path + "/" + SCENE_NAME + f"/graph_visual_keyframe_min1.0_radius0.5_floor{args.floor_idx}.png",
        show_nodes=True, show_edges=True,
        show_keyframes=True,
        draw_kf_yaw=True,
        show_kf_fov=True,     # 开关 FOV
        kf_fov_deg=90.0,      # 你的需求：90°
        kf_fov_range_m=0.5,    # 扇形长度（米），可按相机可见距离调
        show_node_radius=True,
        radius_m=0.0,
        specific_id=988
    )

    """
    可视化采样view的位置
    """
    builder.visualize_view_points_real_robot(
        env=env,
        frames_meta_jsonl_path=args.frames_meta_jsonl,
        out_path=args.memory_path + SCENE_NAME + f"/viz_basic_graph_views_floor{args.floor_idx}.png",
        filter_y_range=floor_range,
        canvas_h=1024,
        show_labels=np.true_divide
    )

    """
    可视化explored和unexplored区域
    """
    # 探索结束后出一张“Habitat topdown + explored覆盖”图
    # floor_min, floor_max = res["ranges_m"][0]

    # draw_explored_on_habitat_topdown(
    #     env=env,
    #     frames_meta_jsonl=args.frames_meta_jsonl,
    #     out_png=os.path.join(args.memory_path, args.scene_name, "topdown_explored_floor0.png"),
    #     y_range=floor_range,  # 用你发现的楼层范围；或 None=全帧
    #     fov_deg=args.fov_deg if hasattr(args,'fov_deg') else 90.0,
    #     res_m=0.05,
    #     ray_stride=4,
    #     max_range_m=10.0,
    #     depth_format="npy",
    #     show_path=True,
    #     draw_frontier=True,
    #     y_floor = floor_center
    # )