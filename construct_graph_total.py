
from scipy.__config__ import show
from scipy._lib.array_api_compat.numpy import False_
from place_graph_builder_obs import PlaceGraphBuilder, SemanticSaver, to_np_rgb3, imread_rgb_uint8, normalize_sem_to_object_list, build_frame_object_features_once
from env import NavEnv, get_objnav_env 
import torch 
import json
import os, math
import numpy as np
import cv2
import sys
import argparse
from tqdm import tqdm
import pickle
import habitat_sim
sys.path.append("/home/wangbo/codes/MG-Nav/third-party/Grounded-SAM-2")

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

##### ，需要用到0.40, 0.20和 不使用ms = min(ms, 3)
SCENE_ID_MAP = {
    # "带前缀的完整ID": "不带前缀的短ID",
    # "00877-4ok3usBNeis": "4ok3usBNeis", "00853-5cdEh9F2hJL": "5cdEh9F2hJL",
    # "00890-6s7QHgap2fW": "6s7QHgap2fW", 
    # "00849-a8BtkwhxdRV": "a8BtkwhxdRV", "00827-BAbdmeyTvMZ": "BAbdmeyTvMZ",
    # "00873-bxsVRursffK": "bxsVRursffK",
    "00810-CrMo8WxCyVb": "CrMo8WxCyVb", 
    # "00891-cvZr5TUy5C5": "cvZr5TUy5C5",
    # "00824-Dd4bFSTQ8gi": "Dd4bFSTQ8gi", "00843-DYehNKdT76V": "DYehNKdT76V",
    # "00821-eF36g7L6Z9M": "eF36g7L6Z9M", "00861-GLAQ4DNUx5U": "GLAQ4DNUx5U",
    # "00815-h1zeeAwLh9Z": "h1zeeAwLh9Z", 
    # "00894-HY1NcmCgn3n": "HY1NcmCgn3n",
    # "00862-LT9Jq6dN3Ea": "LT9Jq6dN3Ea", 
    # "00803-k1cupFYWXJ6": "k1cupFYWXJ6",
    # "00869-MHPLjHsuG27": "MHPLjHsuG27", 
    # "00876-mv2HUxq3B53": "mv2HUxq3B53", "00880-Nfvxx8J5NCo": "Nfvxx8J5NCo",
    # "00814-p53SfW6mjZe": "p53SfW6mjZe", "00835-q3zU7Yy5E5s": "q3zU7Yy5E5s",
    # "00829-QaLdnwvtxbs": "QaLdnwvtxbs",
    # "00832-qyAac8rV8Zk": "qyAac8rV8Zk", "00813-svBbv1Pavdk": "svBbv1Pavdk",
    # "00800-TEEsavR23oF": "TEEsavR23oF", "00871-VBzV5z6i1WS": "VBzV5z6i1WS",
    # "00802-wcojb4TFT35": "wcojb4TFT35", 
    # "00808-y9hTuugGdiq": "y9hTuugGdiq", 
    # "00831-yr17PDCnDDW": "yr17PDCnDDW",
    # "00848-ziup5kvtCCR": "ziup5kvtCCR", 
    # "00839-zt1RVoi7PcG": "zt1RVoi7PcG",
}

##### 这三个场景，在划分floor时，需要用到0.20, 0.08和ms = min(ms, 3)
# SCENE_ID_MAP = {
#     "00820-mL8ThkuaVTM": "mL8ThkuaVTM",
#     "00878-XB4GS9ShBRE": "XB4GS9ShBRE",
# }

##### 这一个场景，在划分floor时，需要用到0.40, 0.20和ms = min(ms, 3)    
# SCENE_ID_MAP = {
#     "00823-7MXmsvcQjpJ": "7MXmsvcQjpJ",
# } 
###### 这两个场景在探索时会少一个island，也就是第0层。 所以需要进行init_satate初始化。同时需要两次单独探索
###### 且需要手动把两次base_floor_y.npy合并，建立新的floor.json。
# SCENE_ID_MAP = {
    # "00844-q5QZSEeHe5g": "q5QZSEeHe5g",
    # "00847-bCPU9suPUw9": "bCPU9suPUw9",
# }

MEMORY_PATH = "memory"
# MEMORY_PATH = "/nas_home/wangbo/vis_nav/844_memory_floor0"
# MEMORY_PATH = "/nas_home/wangbo/vis_nav/847_memory_floor0"
SCENE_NAME = "00877-4ok3usBNeis"

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
parser.add_argument("--scene_name", type=str, default=SCENE_NAME)
parser.add_argument("--scene_dataset_config_file", type=str,
                    default="/nas_dataset/wangbo/HM3D/hm3d_annotated_basis.scene_dataset_config.json")

# === 探索与建图参数 ===
parser.add_argument("--memory_path", type=str, default=MEMORY_PATH)
parser.add_argument("--dino_size", type=str, default="dinov2_vitl14_reg")
parser.add_argument("--random_move_num", type=int, default=30)
parser.add_argument("--floor_idx", type=int, default=1, help="if load_single_floor, single floor idx")

parser.add_argument("--min_dis", type=float, default=1.0, help='FPS min')
parser.add_argument("--radius", type=float, default=0.5, help="node radius")


parser.add_argument("--explore_map",      action="store_true",  default=False)
parser.add_argument("--semantic_analyze", action="store_true",  default=False)
parser.add_argument("--construct_graph",  action="store_true",  default=False)
parser.add_argument("--visualize_graph",  action="store_true",  default=False)



# === 路径派生 ===
parser.add_argument("--npz_path", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "explore_log.npz"))
parser.add_argument("--out_json", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "place_graph_min1.0_radius0.5.json"))
parser.add_argument("--frames_meta_jsonl", type=str,
                    default=os.path.join(MEMORY_PATH, SCENE_NAME, "obs/frames_meta.jsonl"))
parser.add_argument("--sem_jsonl", type=str, default=None)

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
args.predefined_class = ['seating', 'chest of drawers', 'bed', 'bathtub', 'clothes', 'toilet', 'stool', 'sofa', 'sink', 'tv monitor', 'picture', 'cushion', 'towel', 'shower', 'counter', 'fireplace', 'chair', 'table', 'gym equipment', 'cabinet', 'plant']
# args.predefined_class = ["chair", "couch", "potted plant", "bed", "toilet", "tv"]

if not hasattr(args, "device"):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] using {args.device}")

# prepare dinov2
preload_dinov2 = torch.hub.load('/home/wangbo/codes/MG-Nav/third-party/dinov2', args.dino_size, source='local').to('cuda')
# prepare groundedsam2
prelload_gsam2 = GroundedSAM2(
    sam2_checkpoint = "/home/wangbo/codes/MG-Nav/third-party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
    gdino_id = "/home/wangbo/codes/MG-Nav/third-party/Grounded-SAM-2/grounding-dino-tiny",
    default_box_threshold = 0.25,
    default_text_threshold = 0.25,
    device="cuda",
)
# ---------- 主执行逻辑：遍历所有场景 ----------
scene_items = list(SCENE_ID_MAP.items())
for scene_full, scene_short in tqdm(scene_items, desc="Processing scenes", ncols=100):
    # === 🆕 1. 动态更新场景参数 ===
    args.scene_name = scene_full
    args.npz_path = os.path.join(MEMORY_PATH, scene_full, "explore_log.npz")
    args.out_json = os.path.join(MEMORY_PATH, scene_full, f"place_graph_min{args.min_dis}_radius{args.radius}_floor{args.floor_idx}.json")
    args.frames_meta_jsonl = os.path.join(MEMORY_PATH, scene_full, "obs/frames_meta.jsonl")
    args.sem_jsonl = os.path.join(MEMORY_PATH, scene_full, "obs/sem/frames_sem.jsonl")
    os.makedirs(os.path.join(MEMORY_PATH, scene_full), exist_ok=True)

    # # 实例化
    ### 如果是场景00847-bCPU9suPUw9，需要做这个init，因为他有两个island，通常是采集面积大的island，也就上第2，3层；第一层采不到
    ### 因此00847-bCPU9suPUw9需要跑两次，explore 2次，获取两个memory，从而得到完整的map
    # init_state = habitat_sim.AgentState()
    # init_state.position = np.array([3.3501136, -2.5197770, 2.7079227], dtype=np.float32)
    # init_state.rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
    # env = NavEnv(args, init_state=init_state, build_map=False)

    ### 如果是场景00844-q5QZSEeHe5g，需要做这个init，因为他有两个island，通常是采集面积大的island，也就上第2，3层；第一层采不到
    ### 因此00844-q5QZSEeHe5g需要跑两次，explore 2次，获取两个memory，从而得到完整的map
    # init_state = habitat_sim.AgentState()
    # init_state.position = np.array([3.3501136, -2.2461228, 2.7079227], dtype=np.float32)
    # init_state.rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w,x,y,z
    # env = NavEnv(args, init_state=init_state, build_map=False)


    env = NavEnv(args, init_state=None, build_map=False)
    builder = PlaceGraphBuilder(args, preload_dino=preload_dinov2, preload_gsam=prelload_gsam2, env=env)

    # --------------探索的过程中产生的文件------------------
    """
    RGBD 图像: 不变
    explore_log.npz: frame_id, poses_xyz, yaws, features(每一张图的donov2 feature), quats(相机rotation xyz)
    base_height: 每个观测点的y
    """

    if args.explore_map:
        obs = env.reset(args)
        # 如果场景00844-q5QZSEeHe5g，00847-bCPU9suPUw9 需要init_state，因为有两个island，通常是采集面积大的island，也就上第2，3层；第一层采不到
        # obs = env.reset(args, init_state=init_state, build_map=False)
        # # 只探索 & 存数据
        state = env.sims.agents[0].state
        current_island = env.sims.pathfinder.get_island(state.position)
        area_shape = env.sims.pathfinder.island_area(current_island)
        args.random_move_num = int(area_shape / 2) + 1

        # obs = env.reset(args)

        # pf = env.sims.pathfinder

        # def get_all_islands(pf, n_samples=2000):
        #     islands = set()
        #     for _ in range(n_samples):
        #         p = pf.get_random_navigable_point()
        #         isl = pf.get_island(p)
        #         if isl >= 0:
        #             islands.add(isl)
        #     return list(islands)

        # island_ids = get_all_islands(pf, n_samples=2000)
        # areas = [pf.island_area(i) for i in island_ids]
        # largest_island = island_ids[int(np.argmax(areas))]

        # def sample_point_on_island(pf, island_id, max_tries=20000):
        #     for _ in range(max_tries):
        #         p = pf.get_random_navigable_point()
        #         if pf.get_island(p) == island_id:
        #             # 确保在导航网格上
        #             if not pf.is_navigable(p):
        #                 p = pf.snap_point(p)
        #             return p
        #     raise RuntimeError(f"Failed to sample point on island {island_id}")

        # start_pos = sample_point_on_island(pf, largest_island)

        # # ✅ 正确方式：构造 AgentState 再 set
        # agent = env.sims.agents[0]
        # cur = agent.get_state()

        # ns = habitat_sim.AgentState()
        # ns.position = np.array(start_pos, dtype=np.float32)
        # ns.rotation = cur.rotation  # 或者你自己指定的四元数
        # agent.set_state(ns, reset_sensors=True)

        # # 更新岛屿 & 计划探索步数
        # area_shape = pf.island_area(largest_island)
        # args.random_move_num = int(area_shape / 2) + 1
        # print(f"Using largest island {largest_island} with area {area_shape:.2f}")
        print("random_move_num:", args.random_move_num)

        builder.explore(
            env=env,
            random_move_num=args.random_move_num,
            turn_left_deg=args.turn_left,
            lock_floor=False,  # 全场探索也行，后面构图再过滤楼层
            save_rgb="all",  rgb_stride=20,  rgb_format="jpg",
            save_depth="all",   depth_format="npy"  # 全量保留 float32 深度，便于精确后处理
        )
        # builder.explore_frontier(
        #     max_iterations=args.random_move_num,
        #     turn_left_deg=args.turn_left,
        #     save_rgb="all",  rgb_stride=20,  rgb_format="jpg",
        #     save_depth="all",   depth_format="npy"  # 全量保留 float32 深度，便于精确后处理
        # )

    # --------------语义分割中产生的文件------------------
    """
    frames_sem.jsonl: 每一个frame真的frame id, image_wh, class_names和对应的mask, input boxes等
    """

    if args.semantic_analyze:
        semantic_save_dir = os.path.join(args.memory_path, args.scene_name)
        sem_saver = SemanticSaver(semantic_save_dir)
        
        frames_json = os.path.join(semantic_save_dir, "obs", "frames_meta.jsonl")
        exploration_frames = load_frames_from_json(frames_json)
        print(f"共读取 {len(exploration_frames)} 帧。")

        for frame in tqdm(exploration_frames):
            rgb_uint8 = to_np_rgb3(frame["rgb_image"])
            frame_id = frame["frame_id"]
            try:

                prompt = args.predefined_class
                det = prelload_gsam2.detect(
                    rgb=rgb_uint8,
                    text_prompt=prompt
                )
                H, W = rgb_uint8.shape[:2]
                sem_saver.write(frame_id, det, image_wh=(W, H))
                # --- 可选：遇到空检测也记一条轻量日志 ---
                if len(det["class_names"]) == 0:
                    print(f"[INFO] GSAM no-detect at frame {frame_id}")
            except Exception as e:
                import traceback
                print(f"[WARN] GSAM detect failed at frame {frame_id}: {e.__class__.__name__}: {e}")
                traceback.print_exc(limit=1)   # 打印一行栈顶，别太吵
        sem_saver.close()


        # -----------frame object feature extraction----------
        # 逐帧提取所有对象 crop 特征，并存成文件，便于后续的不断建图
        print('node object features extraction:')
        feats_by_frame = {}
        frames_meta = builder.load_frames_meta(args.frames_meta_jsonl)
        sem_dets_by_frame = builder.load_sem_dets(args.sem_jsonl)
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
        save_path = os.path.join(MEMORY_PATH, scene_full, "obs", "sem", "obj_feats_by_frame.pkl")  # 或 .joblib
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_feats_by_frame_pickle(save_path, feats_by_frame)
    # --------------建图中产生的文件------------------
    """
    place_graph.json: memory graph
    obj_feats: instance features [N, D], N是memory graph中存的instance feature。 place_graph中保存的有每一个instance对应的索引
    """

    if args.construct_graph:
        # 查看当前场景的楼层是什么样
        base_height_array = np.load(args.memory_path + "/" + args.scene_name + '/base_height.npy')

        if scene_full == "00844-q5QZSEeHe5g":
            base_height_array_2 = np.load('/nas_home/wangbo/vis_nav/844_memory_floor0/00844-q5QZSEeHe5g' + '/base_height.npy')
            base_height_array = np.concatenate([base_height_array, base_height_array_2])

        if scene_full == "00847-bCPU9suPUw9":
            base_height_array_2 = np.load('/nas_home/wangbo/vis_nav/847_memory_floor0/00847-bCPU9suPUw9' + '/base_height.npy')
            base_height_array = np.concatenate([base_height_array, base_height_array_2])

        res = builder.discover_floors_from_heights_meters(
            base_height_array=base_height_array,   # 已经是“米”
            current_height_m=float(env.agent.get_state().position[1]),
            eps=0.40, min_samples_frac=0.20, percent_clip=(1,99), tiny_margin_m=0.05
        )
        print(res)
        """
        {'num_floors': 2, 'centers_m': [-2.8960976600646973, 0.09855081886053085], 
        'ranges_m': [(-2.949935483932495, -1.3487734206020832), (-1.4487734206020833, 0.15006451606750487)], 
        'current_floor': 0}
        """
        # 保存为 JSON 文件
        floor_data_path = MEMORY_PATH + '/' + args.scene_name + '/floor_data.json' 
        with open(floor_data_path, 'w') as json_file:
            json.dump(res, json_file, indent=4)


        if res["num_floors"] <= 1:
            floor_min, floor_max = None, None
            if args.floor_idx >= 1:
                print(f"this scene has only one floor, we need floor_{args.floor_idx} graph, skip")
                continue
            else:
                args.floor_idx = 0
        else:
            if args.floor_idx >= res["num_floors"]:
                scene_floor_num = res["num_floors"]
                print(f"this scene has only {scene_floor_num} floor, we need floor_{args.floor_idx} graph, skip")
                continue
            else:
                floor_range = res['ranges_m'][args.floor_idx]
                floor_min, floor_max = floor_range

        load_path = os.path.join(MEMORY_PATH, scene_full, "obs", "sem", "obj_feats_by_frame.pkl")
        obj_feats_by_frame = load_feats_by_frame_pickle(load_path)
        # node_radius_m 指范围，min_node_spacing_m是node的之间的最小间距
        builder.build_spatial_node_graph(
                                    npz_path=args.npz_path,
                                    frames_meta_jsonl=args.frames_meta_jsonl,
                                    sem_jsonl=args.sem_jsonl,
                                    node_radius_m=args.radius,
                                    min_node_spacing_m=args.min_dis,
                                    keyframes_per_node=4,
                                    yaw_min_sep_deg=90.0,
                                    y_band=0.6,
                                    floor_min_y=floor_min,
                                    floor_max_y=floor_max,
                                    out_json=args.out_json,
                                    obj_feats_by_frame=obj_feats_by_frame)
        print("Graph construction end!")

    if args.visualize_graph:

        base_height_array = np.load(args.memory_path + "/" + args.scene_name + '/base_height.npy')

        if scene_full == "00844-q5QZSEeHe5g":
            base_height_array_2 = np.load('/nas_home/wangbo/vis_nav/844_memory_floor0/00844-q5QZSEeHe5g' + '/base_height.npy')
            base_height_array = np.concatenate([base_height_array, base_height_array_2])

        if scene_full == "00847-bCPU9suPUw9":
            base_height_array_2 = np.load('/nas_home/wangbo/vis_nav/847_memory_floor0/00847-bCPU9suPUw9' + '/base_height.npy')
            base_height_array = np.concatenate([base_height_array, base_height_array_2])

        res = builder.discover_floors_from_heights_meters(
            base_height_array=base_height_array,   # 已经是“米”
            current_height_m=float(env.agent.get_state().position[1]),
            eps=0.40, min_samples_frac=0.20, percent_clip=(1,99), tiny_margin_m=0.05
        )
        print(res)
        """
        {'num_floors': 2, 'centers_m': [-2.8960976600646973, 0.09855081886053085], 
        'ranges_m': [(-2.949935483932495, -1.3487734206020832), (-1.4487734206020833, 0.15006451606750487)], 
        'current_floor': 0}
        """

        if res["num_floors"] <= 1:
            floor_range = (None, None)
            if args.floor_idx >= 1:
                print(f"this scene has only one floor, we need floor_{args.floor_idx} graph, skip")
                continue
            else:
                args.floor_idx = 0
            if res["num_floors"] == 0:
                floor_center = base_height_array[0]
            else:
                floor_center = res['centers_m'][args.floor_idx]
        else:
            if args.floor_idx >= res["num_floors"]:
                scene_floor_num = res["num_floors"]
                print(f"this scene has only {scene_floor_num} floor, we need floor_{args.floor_idx} graph, skip")
                continue
            else:
                floor_center = res['centers_m'][args.floor_idx]
                floor_range = res['ranges_m'][args.floor_idx]

        """
        可视化构建的graph
        """
        # 假设 builder 是 PlaceGraphBuilder 实例，env 是已初始化的 Habitat NavEnv
        json_path = args.out_json

        builder.visualize_graph_from_json(
            env=env,
            graph_json_path=json_path,
            out_path=args.memory_path + "/" + args.scene_name + f"/graph_visual_min{args.min_dis}_radius{args.radius}_floor{args.floor_idx}.png",
            show_nodes=True,
            show_edges=True,
            edge_types=None,         # 仅画 temporal 就写 ["temporal"]
            draw_node_ids=True,
            canvas_h=1024,
            floor_y=floor_center,
        )

        builder.visualize_graph_from_json_with_keyframes(
            env, 
            graph_json_path=json_path, 
            out_path=args.memory_path + "/" + args.scene_name + f"/graph_visual_keyframe_min{args.min_dis}_radius{args.radius}_floor{args.floor_idx}.png",
            show_nodes=True, show_edges=True,
            show_keyframes=True,
            draw_kf_yaw=True,
            show_kf_fov=True,     # 开关 FOV
            kf_fov_deg=90.0,      # 你的需求：90°
            kf_fov_range_m=0.5,    # 扇形长度（米），可按相机可见距离调
            floor_y=floor_center,
            show_node_radius=True,
            radius_m=0.0,
            specific_id=0
        )

        """
        可视化图，不包括topdown map，只画节点和边，存成pdf。
        """

        #   
        # builder.visualize_graph_from_json_with_keyframes_pdf(
        #     env, 
        #     graph_json_path=json_path, 
        #     out_pdf_path=args.memory_path + "/" + args.scene_name + f"/trans_graph_visual_keyframe_min{args.min_dis}_radius{args.radius}_floor{args.floor_idx}.pdf",
        #     show_nodes=True, show_edges=True,
        #     show_keyframes=True,
        #     draw_kf_yaw=True,
        #     show_kf_fov=True,     # 开关 FOV
        #     kf_fov_deg=90.0,      # 你的需求：90°
        #     kf_fov_range_m=0.5,    # 扇形长度（米），可按相机可见距离调
        #     floor_y=floor_center,
        #     show_node_radius=True,
        #     radius_m=0.0,
        #     specific_id=0
        # )

        """
        # 可视化采样view的位置
        """
        builder.visualize_view_points(
            env=env,
            frames_meta_jsonl_path=args.frames_meta_jsonl,
            out_path=args.memory_path + "/" + args.scene_name + f"/viz_basic_graph_views_floor{args.floor_idx}.png",
            filter_y_range=floor_range,
            canvas_h=1024,
            y_floor=floor_center,
            show_labels=np.true_divide
        )

        """
        可视化采样轨迹和采样view，并存成pdf
        """
        # builder.visualize_view_points_traj(
        #     env=env,
        #     frames_meta_jsonl_path=args.frames_meta_jsonl,
        #     out_path=args.memory_path + "/" + args.scene_name + f"/traj_points_floor{args.floor_idx}.pdf",
        #     point_radius=3,
        #     # selected_subgoal_ids=[98,104,105,114,119,120,128,129,131,132,136],
        #     selected_subgoal_ids=[3,15,32,40,130,131,136,16],
        #     traj_thickness=2,
        #     filter_y_range=floor_range,
        #     y_floor=floor_center,
        # )
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

    # 存在memory泄漏
    del builder
    if env is not None:
        try:
            env.close()
        except:
            pass
        del env
    torch.cuda.empty_cache()
    import gc; gc.collect()