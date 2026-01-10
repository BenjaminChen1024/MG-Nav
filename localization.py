# imagenav_localizer_min.py
# -*- coding: utf-8 -*-
import os
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal, Any

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
import pycocotools.mask as mask_util

import cv2  
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height, to_grid  
import matplotlib.pyplot as plt

def _to_np_rgb3(x: np.ndarray) -> np.ndarray:
    # habitat 可能给 RGBA 或 float
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    if x.shape[-1] == 4:
        x = x[..., :3]
    return x

class DinoGlobalEncoder:
    """Minimal global encoder using DINOv2 via torch.hub."""
    def __init__(self, pre_load_dinov2, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = pre_load_dinov2
        self.prep = T.Compose([
            T.ToPILImage(),
            T.Resize(518, antialias=True),
            T.CenterCrop(518),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    @torch.inference_mode()
    def encode(self, rgb: np.ndarray) -> np.ndarray:
        rgb = _to_np_rgb3(rgb)
        x = self.prep(rgb).unsqueeze(0).to(self.device)
        feat = self.model(x)  # [1, D]
        feat = F.normalize(feat, dim=-1).cpu().numpy()[0].astype(np.float32)
        return feat

    @torch.inference_mode()
    def extract_patch_tokens(self, rgb: np.ndarray):
        """
        返回 tokens_grid: (Hp, Wp, D) 以及 (Hp, Wp)
        预处理分辨率固定 518×518，ViT/14 -> 37×37 网格。
        """
        rgb = _to_np_rgb3(rgb)
        x = self.prep(rgb).unsqueeze(0).to(self.device)           # [1,3,518,518]
        # 取最后一层的 patch tokens（不含 cls），形状 [1, N, D] -> [N, D]
        tokens = self.model.get_intermediate_layers(x, n=1, return_class_token=False)[0][0]  # [N, D]

        # 由 patch size 推 Hp, Wp
        ps = self.model.patch_embed.patch_size
        ps = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        Hp = Wp = int(round(518 / ps))                            # 518/14 = 37
        tokens = tokens[: Hp*Wp].reshape(Hp, Wp, -1)              # [Hp, Wp, D]
        return tokens.detach().cpu().numpy().astype(np.float32), (Hp, Wp)



# ========== 图数据结构（仅 rep_npz_idx / rep_frame_id / members） ==========
@dataclass
class NodeLite:
    id: int
    center: Dict[str, float]                    # {"x","y","z"}
    radius: float
    members: List[int]       # 该节点覆盖的 frame_ids
    keyframes: List[int]     # 该节点的 keyframe frame_ids
    object_list: List[Dict[str, Any]] 


@dataclass
class EdgeLite:
    u: int
    v: int
    dist: float
    type: str = "temporal"
    delta_pose: Dict[str, float] = field(default_factory=dict)
    count: int = 1


class PlaceGraph:
    def __init__(self, graph_json: str, explore_npz: str):
        with open(graph_json, "r") as f:
            G = json.load(f)

        # === 新增：读 meta / features ===
        self.meta      = G.get("meta", {}) or {}
        self.features  = G.get("features", {}) or {}
        self._npz_path = self.meta.get("npz_path", explore_npz)   # 允许用参数兜底
        self._obj_feats_path = self.features.get("obj_feats_path")
        self._feat_dim = int(self.features.get("feat_dim", 0))

        # 节点（不读取 global_feat）
        self.nodes: List[NodeLite] = []
        for nd in G["nodes"]:
            center = (float(nd["center"]["x"]), float(nd["center"]["y"]),
                    float(nd["center"]["z"]))
            members = [int(x) for x in nd.get("members", [])]
            obj_list = nd.get("object_list", []) or []
            key_frames = [int(x) for x in nd.get("keyframes", [])]

            self.nodes.append(NodeLite(
                id=int(nd["id"]), 
                center=center,
                radius=nd["radius"],
                members=members,
                keyframes=key_frames,
                object_list=obj_list
            ))

        # 边 → 无向邻接
        self.edges: List[EdgeLite] = []
        for e in G["edges"]:
            u, v = int(e["source_id"]), int(e["target_id"])
            d = float(e.get("distance", 0.0))
            self.edges.append(EdgeLite(u=u, v=v, dist=d))

        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        for e in self.edges:
            self.adj.setdefault(e.u, []).append((e.v, e.dist))
            self.adj.setdefault(e.v, []).append((e.u, e.dist))


        ## rgb图片的读取，便于后续进行object feature 对应
        self._frame_to_rgb_path: Dict[int, str] = {}
        root_dir = os.path.dirname(explore_npz)
        meta_path = os.path.join(root_dir, "obs", "frames_meta.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as mf:
                for line in mf:
                    rec = json.loads(line)
                    fid = int(rec["frame_id"])
                    rp  = rec.get("rgb_path", None)
                    if rp:
                        self._frame_to_rgb_path[fid] = rp

    def load_rgb_by_frame_id(self, frame_id: int) -> np.ndarray:
        """返回该 frame_id 的 RGB uint8(H,W,3)。优先用 frames_meta.jsonl 里的路径。"""

        p = self._frame_to_rgb_path.get(int(frame_id)) if hasattr(self, "_frame_to_rgb_path") else None
        if p and os.path.exists(p):
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ========== A* 规划 ==========
class AStarPlanner:
    def __init__(self, graph: PlaceGraph):
        self.G = graph
        # —— 正确的 id -> idx 映射，用于取坐标
        self.id2idx = {n.id: i for i, n in enumerate(self.G.nodes)}
        # 节点坐标表（按列表顺序）
        self.XZ = np.array([[n.center[0], n.center[2]] for n in self.G.nodes], dtype=np.float32)
        # 规范化加权邻接：缺省/非正距离视为 1
        self.adjW = {}
        for uid, nbrs in self.G.adj.items():
            row = []
            for vid, dist in nbrs:
                w = float(dist) if (dist is not None and float(dist) > 0) else 1.0
                row.append((vid, w))
            self.adjW[uid] = row

    # 启发函数（用 id -> idx 取坐标）
    def _h_ids(self, a_id: int, b_id: int) -> float:
        ia = self.id2idx.get(a_id, None)
        ib = self.id2idx.get(b_id, None)
        if ia is None or ib is None:
            return 0.0
        dx, dz = self.XZ[ia] - self.XZ[ib]
        return float(np.hypot(dx, dz))

    # —— 无权最短路（BFS）
    def _bfs_hop_path(self, src_id: int, dst_id: int):
        from collections import deque
        if src_id == dst_id:
            return [src_id]
        q = deque([src_id])
        prev = {src_id: None}
        while q:
            u = q.popleft()
            for v, _w in self.adjW.get(u, []):
                if v in prev:  # visited
                    continue
                prev[v] = u
                if v == dst_id:
                    # 回溯
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
        return []  # 不连通

    # —— Dijkstra（加权）
    def _dijkstra(self, src_id: int, dst_id: int):
        import heapq
        if src_id == dst_id:
            return [src_id]
        pq = [(0.0, src_id, None)]  # (g, node, parent)
        best_g = {src_id: 0.0}
        parent = {}
        vis = set()
        while pq:
            g, u, p = heapq.heappop(pq)
            if u in vis:
                continue
            vis.add(u)
            parent[u] = p
            if u == dst_id:
                path = [u]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            for v, w in self.adjW.get(u, []):
                ng = g + float(w)
                if v not in best_g or ng < best_g[v] - 1e-9:
                    best_g[v] = ng
                    heapq.heappush(pq, (ng, v, u))
        return []

    # —— A*（加权）
    def _a_star(self, src_id: int, dst_id: int):
        import heapq
        if src_id == dst_id:
            return [src_id]
        pq = [(self._h_ids(src_id, dst_id), 0.0, src_id, None)]  # (f,g,u,parent)
        best_g = {src_id: 0.0}
        parent = {}
        vis = set()
        while pq:
            f, g, u, p = heapq.heappop(pq)
            if u in vis:
                continue
            vis.add(u)
            parent[u] = p
            if u == dst_id:
                path = [u]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            for v, w in self.adjW.get(u, []):
                ng = g + float(w)
                if v not in best_g or ng < best_g[v] - 1e-9:
                    best_g[v] = ng
                    fv = ng + self._h_ids(v, dst_id)
                    heapq.heappush(pq, (fv, ng, v, u))
        return []

    # —— 对外接口：优先保证“能到达”，再追求“最短”
    def shortest_path(self, src_id: int, dst_id: int):
        # 先用 BFS 检查连通 & 取一个兜底路径
        bfs_path = self._bfs_hop_path(src_id, dst_id)
        if not bfs_path:
            return []  # 不连通

        # 能到达的话，试 A*
        a_path = self._a_star(src_id, dst_id)
        if a_path:
            return a_path

        # A* 失败就退 Dijkstra
        d_path = self._dijkstra(src_id, dst_id)
        if d_path:
            return d_path

        # 还不行就用 BFS（最少跳数，不保证距离最短）
        return bfs_path


# ========== Retrieve ==========
class GraphLocalizer:
    def __init__(self, graph: PlaceGraph, encoder: DinoGlobalEncoder):
        self.G = graph
        self.encoder = encoder

        self._npz_path = self.G._npz_path
        self.features  = self.G.features
        self._obj_feats_path= self.G._obj_feats_path
        self._feat_dim =  self.G._feat_dim

        # 缓存
        self._frameid2feat = None     # {frame_id: feature}
        self._obj_feats = None        # np.ndarray[F, D]
        self._node_objfeat_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._node_objfeat_built: Dict[int, bool] = {}


        # 每个 node 的 keyframe 特征矩阵缓存
        self._node_kf_feats: Dict[int, np.ndarray] = None  # nid -> [K,D] (L2)

        # 预备载入
        self._ensure_frameid2feat()
        self._ensure_node_keyframe_tables()
        self._ensure_obj_feats()


    @staticmethod
    def _ensure_uint8(rgb) -> np.ndarray:
        if rgb.dtype == np.uint8:
            arr = rgb
        else:
            arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr

    def _ensure_frameid2feat(self):
            if self._frameid2feat is not None:
                return
            self._frameid2feat = {}
            if not self._npz_path or (not os.path.exists(self._npz_path)):
                return
            data = np.load(self._npz_path, allow_pickle=True)
            fids  = data["frame_ids"].astype(np.int64)
            feats = data["feats"].astype(np.float32)
            feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
            for i, fid in enumerate(fids):
                self._frameid2feat[int(fid)] = feats[i]

    def _ensure_obj_feats(self):
        if self._obj_feats is not None:
            return
        if not self._obj_feats_path or (not os.path.exists(self._obj_feats_path)):
            self._obj_feats = None
            return
        X = np.load(self._obj_feats_path).astype(np.float32)
        self._obj_feats = X

    def _ensure_node_keyframe_tables(self):
        if self._node_kf_feats is not None:
            return
        self._node_kf_feats = {}
        for nid, n in enumerate(self.G.nodes):
            fids = list(getattr(n, "keyframes", []) or [])
            vecs = []
            for fid in fids:
                v = self._frameid2feat.get(int(fid))
                if v is not None:
                    vecs.append(v)
            if vecs:
                M = np.stack(vecs, 0).astype(np.float32)   # [K,D]
            else:
                M = np.empty((0, self._feat_dim), np.float32)
            self._node_kf_feats[nid] = M
    
    #################
    ##### 通过global dinov2 feature来retrieve
    #################
    # ---------- 用 keyframe 特征给所有节点打分 ----------
    def _score_nodes_by_keyframes(self, q_vec: np.ndarray, agg: str = "max", topm: int = 2) -> np.ndarray:
        """
        q_vec: [D]（L2）
        返回：sims_g[n_nodes]，每个节点 = 聚合(keyframe_i · q)
        agg: "max" | "mean" | "topm_mean"
        """
        M = len(self.G.nodes)
        sims_g = np.full((M,), -1e9, np.float32)  # 用 -inf 代表无特征
        for nid in range(M):
            K = self._node_kf_feats[nid]
            if K.size == 0:
                continue
            s = (K @ q_vec)  # [K]
            if agg == "mean":
                sims_g[nid] = float(np.mean(s))
            elif agg == "topm_mean":
                m = min(topm, s.shape[0])
                sims_g[nid] = float(np.mean(np.sort(s)[-m:]))
            else:  # "max"（默认）
                sims_g[nid] = float(np.max(s))
        return sims_g

    def localize(self, rgb: np.ndarray, topk: int = 5,
                    kf_agg: str = "max", kf_topm: int = 2) -> Tuple[int, List[Tuple[int, float]]]:
            q = self.encoder.encode(self._ensure_uint8(rgb)).astype(np.float32)
            q /= (np.linalg.norm(q) + 1e-9)
            sims_g = self._score_nodes_by_keyframes(q, agg=kf_agg, topm=kf_topm)
            order = np.argsort(-sims_g)[:max(1, topk)]
            top = [(int(self.G.nodes[i].id), float(sims_g[i]),
                    np.asarray(self.G.nodes[i].center, dtype=np.float32)
                ) for i in order]
            return top[0][0], top
    #################
    ##### retrieve with global feature and instance feature
    #################

    # ============ 新增：RLE -> mask ============
    @staticmethod
    def _rle_to_bool_mask(rle: dict) -> np.ndarray:
        m = mask_util.decode(rle)  # (H,W,1) 或 (H,W)
        if m.ndim == 3:
            m = m[:, :, 0]
        return m.astype(bool)

    @staticmethod
    def _downweight_mask_to_grid(mask_bool: np.ndarray, Hp: int, Wp: int) -> torch.Tensor:
        """
        mask(H,W){0,1} -> 软权重网格(Hp,Wp) in [0,1], 用于对 patch-tokens 做加权池化。
        """
        m = torch.from_numpy(mask_bool.astype(np.float32))  # [H,W]
        m = torch.nn.functional.interpolate(
            m[None, None, :, :], size=(Hp, Wp), mode="bilinear", align_corners=False
        )[0, 0]  # [Hp, Wp]
        return m

    def _masked_dino_feature(self, rgb_uint8: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:

        self.encoder.model.eval()
        with torch.no_grad():
            x = self.encoder.prep(self._ensure_uint8(rgb_uint8)).unsqueeze(0).to(self.encoder.device)  # [1,3,H,W]

            out = self.encoder.model.get_intermediate_layers(x, n=1, return_class_token=False)[0]  # [1, N, D]
            tokens = out[0]  # [N,D]，不含 cls token
        N, D = tokens.shape

        # 2) 网格尺寸 (Hp, Wp)
        # 常见作法：已知输入固定 518 -> patch=14/16 之类；更稳的是根据特征图形状推回网格
        # 这里先用近似：找最接近方阵的 (Hp, Wp)
        w = int(round((N) ** 0.5))
        h = max(1, int(round(N / max(1, w))))
        while h * w < N:  # 兜底补到 >=N
            w += 1
        Hp, Wp = h, w
        tokens = tokens[: Hp * Wp, :].reshape(Hp, Wp, D)  # [Hp,Wp,D]

        # 3) mask 下采样到网格
        m = self._downweight_mask_to_grid(mask_bool, Hp, Wp)  # [Hp,Wp] in [0,1]

        # 4) 加权平均池化
        wgt = (m > 0.2).float() * m
        wgt = wgt / (wgt.sum() + 1e-6)
        wgt = wgt.to(tokens.device)
        feat = (tokens * wgt[..., None]).sum(dim=(0, 1))  # [D]
        feat = feat / (feat.norm() + 1e-6)
        return feat.cpu().numpy().astype(np.float32)

    def _build_node_object_features_once(self, node_id: int) -> None:
        """
        从 node.object_list 里读取每个实例的单个 feature index，
        直接在 self._obj_feats 中取出向量，按 class_name 聚合：
        self._node_objfeat_cache[node_id] = {class_name: [feat_obj, ...], ...}
        """
        if self._node_objfeat_built.get(node_id, False):
            return

        obj_map: Dict[str, List[np.ndarray]] = {}
        n = self.G.nodes[node_id]
        olist = getattr(n, "object_list", None) or []

        if (self._obj_feats is None) or (self._obj_feats.size == 0) or (not olist):
            self._node_objfeat_cache[node_id] = {}
            self._node_objfeat_built[node_id] = True
            return

        N = self._obj_feats.shape[0]
        for obj in olist:
            cls_name = str(obj.get("class_name", "")).strip().lower()
            idx_any  = obj.get("feature_indices", None)
            if idx_any is None:
                continue

            # 兼容 int 或 [int]
            if isinstance(idx_any, (list, tuple)):
                if not idx_any:
                    continue
                idx0 = int(idx_any[0])
            else:
                idx0 = int(idx_any)

            # 越界保护
            if idx0 < 0 or idx0 >= N:
                continue

            v = self._obj_feats[idx0]  # [D]
            # 保险起见再做一次 L2（如果写盘时已 L2，这里无害）
            v = v / (np.linalg.norm(v) + 1e-9)
            obj_map.setdefault(cls_name, []).append(v.astype(np.float32))

        self._node_objfeat_cache[node_id] = obj_map
        self._node_objfeat_built[node_id] = True


    def _query_object_features(self, rgb_uint8: np.ndarray, det: dict) -> Dict[str, List[np.ndarray]]:
        """
        从当前帧的 GSAM 结果 'det'（包含 class_names + RLE masks）抽取查询对象特征。
        return { class_name: [feat_obj, ...], ... }
        Example: out = {"sofa": [feat_sofa1, feat_sofa2, feat_sofa3, ..], "table": [feat_table1], "chair": [feat_chair1, feat_chair2]}
        """
        cls = (det or {}).get("class_names", []) or []
        rles = (det or {}).get("masks", []) or []
        out: Dict[str, List[np.ndarray]] = {}
        for cname, rle in zip(cls, rles):
            try:
                mask = self._rle_to_bool_mask(rle)
                feat = self._masked_dino_feature(rgb_uint8, mask)
                out.setdefault(str(cname).lower(), []).append(feat)
            except Exception:
                continue
        return out


    def mean_of_per_instance_best(
        self,
        q_map: dict[str, list[np.ndarray]],   # 待定位帧：类名 -> [feat_i...]，每个 feat 已 L2 归一化，shape[D]
        n_map: dict[str, list[np.ndarray]],   # 节点：同上
        missing_penalty: float = 0.0          # 节点里没这个类时，该实例分数给多少
    ) -> float:
        """对 query 的每个实例，找节点中同类里最相似的实例分数（余弦），最后取平均。"""
        per_inst_scores = []

        for cls, q_feats in q_map.items():
            # 节点没有该类：要么记 0，要么跳过（改下面一行即可）
            n_feats = n_map.get(cls)
            if not n_feats:
                per_inst_scores.extend([missing_penalty] * len(q_feats))
                continue

            N = np.stack(n_feats, axis=0)    # [nn, D] ，已归一化
            for q in q_feats:
                # q: [D]，已归一化
                s = (N @ q).max()            # 该实例对该类中“最像”的一个
                per_inst_scores.append(float(s))

        if not per_inst_scores:
            return 0.0
        return float(np.mean(per_inst_scores))

    def localize_with_instance(
        self,
        rgb: np.ndarray,
        det: dict, # rgb的分割结果
        topk: int = 5,
        alpha_global: float = 0.4,
        coarse_k: int = 20,            # 新增：全局粗检候选数
        kf_agg: str = "max",
        kf_topm: int = 2
    ):
        """
        两阶段：先用全局特征取 Top-K 候选，再在候选里用实例匹配( mean-of-per-instance-best )重排。
        返回：(best_node_id, top_list)，top_list = [(node_id, fused_score, {"global":..., "obj":...}), ...]
        """
        # ---------- Stage 1: 全局粗检 ----------
        q = self.encoder.encode(self._ensure_uint8(rgb)).astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        sims_g = self._score_nodes_by_keyframes(q, agg=kf_agg, topm=kf_topm)

        K = max(topk, coarse_k)
        cand_idx = np.argsort(-sims_g)[:K]  # 只在候选里做实例重排

        # Stage 2：实例重排（保持你的新图格式：obj_feats.npy + feature_indices）
        rgb_u8 = self._ensure_uint8(rgb)
        q_objs = self._query_object_features(rgb_u8, det)
        if not q_objs:
            order = cand_idx[:max(1, topk)]
            top = [(int(self.G.nodes[i].id), float(sims_g[i]),
                    {"global": float(sims_g[i]), "obj": 0.0}) for i in order]
            return top[0][0], top

        obj_sims = np.zeros_like(sims_g)
        for nid in cand_idx:
            self._build_node_object_features_once(int(nid))
            nmap = self._node_objfeat_cache.get(int(nid), {})
            obj_sims[nid] = self.mean_of_per_instance_best(q_map=q_objs, n_map=nmap, missing_penalty=0.0)

        fused = alpha_global * sims_g + (1.0 - alpha_global) * obj_sims
        order = cand_idx[np.argsort(-fused[cand_idx])][:max(1, topk)]
        top = [(int(self.G.nodes[i].id),
                float(fused[i]),
                {"global": float(sims_g[i]), "obj": float(obj_sims[i])},
                np.asarray(self.G.nodes[i].center, dtype=np.float32)
        ) for i in order]
        return top[0][0], top

# ========== 机器人封装 ==========
class ImageNavGraphRobot:
    """
    - 读图（JSON+NPZ）
    - 从 obs 里取 start / goal（imagegoal/instance_imagegoal）做本地化
    - 用 A* 在图上规划节点路径
    - 预留：导出 waypoint poses 给下游控制器（如 NavDP）
    """
    def __init__(self, args, graph_json: str, explore_npz: str, preload_gsam,
                 preload_dino: str = "dinov2_vitl14", ):
        self.graph = PlaceGraph(graph_json, explore_npz)
        self.args = args

        # DINOv2 encoder
        if preload_dino is None:
            dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
            self.encoder = DinoGlobalEncoder(pre_load_dinov2=dinov2, device=args.device)
        else:
            self.encoder = DinoGlobalEncoder(pre_load_dinov2=preload_dino, device=args.device)
        
        # GroundedSam2
        self._gsam2 = preload_gsam
        self.obj_text_prompt = args.predefined_class


        self.localizer = GraphLocalizer(self.graph, self.encoder)
        self.planner = AStarPlanner(self.graph)
        
        # 可选：在 args 里传 floor_json / floor_idx / y_pad_m
        self.floor_filter: Optional[Tuple[float, float]] = None  # (ymin, ymax)
        if getattr(args, "floor_json", None):
            self.set_floor_filter_from_json(args.floor_json, getattr(args, "floor_idx", None))

    def set_floor_filter_from_json(self, floor_json_path, idx=None):
        """读取 floor_data.json 并设置楼层过滤范围（带兜底逻辑）
        约定：当 idx 指向最顶层时，仅设置下限（上限为 None）。
        """
        try:
            with open(floor_json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load floor_json ({floor_json_path}): {e}, skip floor filtering.")
            self.floor_filter = (None, None)
            return

        ranges = data.get("ranges_m", [])
        if not ranges or len(ranges) == 0:
            print(f"[WARN] Empty floor_json ({floor_json_path}), fallback to single floor mode.")
            # 给一个默认范围（例如 ±10 米，假设是单层环境）
            floor_min_y = -100.0
            floor_max_y = 100.0
            self.floor_filter = (floor_min_y, floor_max_y)
            return

        # 如果 idx 超出范围，则取第一个
        if idx is None or idx >= len(ranges) or idx < 0:
            print(f"[INFO] floor_idx={idx} invalid, using default=0")
            idx = 0

        floor_min, floor_max = float(ranges[idx][0]), float(ranges[idx][1])

        # 如果是最顶层：仅设置下限
        if idx == len(ranges) - 1:
            self.floor_filter = (floor_min, 1000.0)
            print(f"[FloorFilter] use top floor={idx} y∈[{floor_min:.3f}, 1000.0)")
        # 如果是第一层（最低层）：仅设置上限
        elif idx == 0:
            self.floor_filter = (-1000.0, floor_max) # 使用一个非常小的数作为下限
            print(f"[FloorFilter] use bottom floor={idx} y∈[-1000.0, {floor_max:.3f}]")
        # 否则：设置正常的上下限
        else:
            self.floor_filter = (floor_min, floor_max)
            print(f"[FloorFilter] use floor={idx} y∈[{floor_min:.3f}, {floor_max:.3f}]")


    def on_graph_floor(self, y: float) -> bool:
        """y 是否在当前楼层范围内。若未设置过滤，默认 True。"""
        if self.floor_filter is None:
            return True
        ymin, ymax = self.floor_filter
        return (ymin <= float(y) <= ymax)
    ###############################
    # locoalization
    ###############################
    def get_true_start_goal_positions(self, env):
        """
        返回 (start_xyz, goal_xyz)，都为 np.ndarray(3,)
        - start_xyz：当前 agent 的世界坐标
        - goal_xyz ：episode 的 image-goal 对应 view_point 的世界坐标
        """
        ep = env.current_episode
         # 兼容不同 habitat 版本的取位姿 API
        try:
            start_xyz = np.asarray(env.sim.get_agent_state(0).position, dtype=np.float32)
        except Exception:
            start_xyz = np.asarray(env.sim.get_agent(0).get_state().position, dtype=np.float32)

        # ---- 优先: Instance-ImageGoal (episode 内含 goal_image_id & view_points) ----
        gid = getattr(ep, "goal_image_id", None)
        view_points = getattr(ep.goals[0], "view_points", None)

        if (gid is not None) and (view_points is not None) and len(view_points) > 0:
            # 防越界
            if not (0 <= int(gid) < len(view_points)):
                # gid 异常时，退回 0（也可选最近视点）
                gid = 0
            goal_xyz = np.asarray(view_points[int(gid)].agent_state.position, dtype=np.float32)
            return start_xyz, goal_xyz

        # ---- 回退: ImageGoal（只有 goals[0].position）----
        goal_pos = getattr(ep.goals[0], "position", None)
        assert goal_pos is not None, "Episode has no goals[0].position!"
        goal_xyz = np.asarray(goal_pos, dtype=np.float32)
        return start_xyz, goal_xyz

    def localize_start_goal_from_obs(self, obs: Dict, env) -> Tuple[int, int, List[Tuple[int, float]], List[Tuple[int, float]]]:

        start_xyz, goal_xyz = self.get_true_start_goal_positions(env)

        start_rgb = obs["rgb"]
        _, start_top = self.localizer.localize(start_rgb, topk=5, kf_agg="max", kf_topm=2)

        goal_rgb = obs.get("instance_imagegoal", None)
        if goal_rgb is None:
            goal_rgb = obs.get("imagegoal", None)
        _, goal_top = self.localizer.localize(goal_rgb, topk=5, kf_agg="max", kf_topm=2)

        return start_xyz, goal_xyz, start_top, goal_top

    def localize_start_goal_from_obs_with_instance(self, obs: Dict, env) -> Tuple[int, int, List[Tuple[int, float]], List[Tuple[int, float]]]:
        prompt = self.obj_text_prompt
        
        start_xyz, goal_xyz = self.get_true_start_goal_positions(env)   
        start_rgb = obs["rgb"]

        det_start = self._gsam2.detect(
            rgb=start_rgb,
            text_prompt=prompt
        )
        _, start_top = self.localizer.localize_with_instance(start_rgb, det_start, topk=5, alpha_global=0.3, coarse_k=50, kf_agg="max", kf_topm=2)

        goal_rgb = obs.get("instance_imagegoal", None)
        if goal_rgb is None:
            goal_rgb = obs.get("imagegoal", None)

        det_goal = self._gsam2.detect(
            rgb=goal_rgb,
            text_prompt=prompt
        )
        _, goal_top = self.localizer.localize_with_instance(goal_rgb, det_goal, topk=5, alpha_global=0.3, coarse_k=50, kf_agg="max", kf_topm=2)

        return start_xyz, goal_xyz, start_top, goal_top

    def localize_obs(self, obs: Dict, env) -> Tuple[int, int, List[Tuple[int, float]], List[Tuple[int, float]]]:

        rgb = obs["rgb"]
        _, top = self.localizer.localize(rgb, topk=5, kf_agg="max", kf_topm=2)

        return top

    def localize_obs_with_instance(self, obs: Dict, env) -> Tuple[int, int, List[Tuple[int, float]], List[Tuple[int, float]]]:
        prompt = self.obj_text_prompt
        
        rgb = obs["rgb"]

        det_start = self._gsam2.detect(
            rgb=rgb,
            text_prompt=prompt
        )
        _, top = self.localizer.localize_with_instance(rgb, det_start, topk=5, alpha_global=0.3, coarse_k=50, kf_agg="max", kf_topm=2)

        return top

    ###############################
    # Path Plan
    ###############################
    def plan_waypoints(self, src_node_id: int, dst_node_id: int) -> List[int]:
        return self.planner.shortest_path(src_node_id, dst_node_id)

    def waypoints_pose(self, node_path: List[int]) -> List[Dict[str, float]]:
        id2node = {n.id: n for n in self.graph.nodes}
        poses = []
        for nid in node_path:
            n = id2node[nid]
            poses.append({"x": n.center[0], "y": n.center[1], "z": n.center[2]})
        return poses

    def plan_waypoints_with_true_points(
        self,
        true_start_xyz,                   # (x,y,z)
        retrieved_start_node_id: int,     # e.g. start_top[0][0]
        retrieved_goal_node_id: int,      # e.g. goal_top[0][0]
        true_goal_xyz=None,               # (x,y,z) or None
        use_bfs_fallback: bool = True,    # 允许 BFS 兜底
    ):
        id2node = {n.id: n for n in self.graph.nodes}

        # ---- 1) 真实起点 -> 检索起点节点 的直连段 ----
        sx, sz = float(true_start_xyz[0]), float(true_start_xyz[2])
        if retrieved_start_node_id not in id2node:
            # 起点 id 无效，直接返回只有 prefix（退化为 0 长度，避免崩）
            prefix = ((sx, sz), (sx, sz))
            return {
                "node_path": [],
                "prefix_segment": prefix,
                "suffix_segment": None,
                "reason": f"start node {retrieved_start_node_id} not in graph",
            }
        sn = id2node[retrieved_start_node_id]
        prefix = ((sx, sz), (float(sn.center[0]), float(sn.center[2])))

        # ---- 2) 图上路径：A* -> BFS 兜底 ----
        node_path = self.planner.shortest_path(retrieved_start_node_id, retrieved_goal_node_id)
        if (not node_path) and use_bfs_fallback and hasattr(self.planner, "any_path_bfs"):
            node_path = self.planner.any_path_bfs(retrieved_start_node_id, retrieved_goal_node_id)

        # ---- 3) 真实终点直连段（可选） ----
        suffix = None
        if true_goal_xyz is not None:
            gx, gz = float(true_goal_xyz[0]), float(true_goal_xyz[2])

            if node_path:
                # 用图上“最后一个节点”去连真实 goal
                last_node_id = node_path[-1]
                if last_node_id in id2node:
                    ln = id2node[last_node_id]
                    suffix = ((float(ln.center[0]), float(ln.center[2])), (gx, gz))
                else:
                    # 理论不会发生；做个保险
                    suffix = ((float(sn.center[0]), float(sn.center[2])), (gx, gz))
            else:
                # 图上无路径：直接从“检索起点节点”连到真实 goal
                suffix = ((float(sn.center[0]), float(sn.center[2])), (gx, gz))

        return {
            "node_path": node_path,
            "prefix_segment": prefix,
            "suffix_segment": suffix,
            "reason": None if node_path else "no graph path found; returned straight segments only",
        }
    ###############################
    # 可视化
    ###############################
    # 世界坐标(x,z) -> topdown像素(col,row)
    @staticmethod
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

    @staticmethod
    def _draw_dot(img, p, color, r=4, thickness=-1):
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def _draw_text(img, p, text, color=(255, 255, 255)):
        cv2.putText(img, text, (int(p[0]) + 4, int(p[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def _maybe_goal_px_from_td(td: dict):
        """从topdown measurement里尝试读取目标像素点（不同配置字段名可能不同）"""
        for k in ["goal_map_points", "goal_positions", "goal_map_coords"]:
            if k in td and isinstance(td[k], (list, tuple)) and len(td[k]) > 0:
                coords = []
                for c in td[k]:
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        coords.append((int(c[0]), int(c[1])))
                if coords:
                    return coords
        return None

    def _prepare_canvas_and_nodes(self, env, canvas_h: int):
        """生成底图(vis)、缩放比例(scale)、节点像素(node_px)、id->idx映射"""
        metrics = env.get_metrics() or {}
        td = metrics.get("top_down_map", None)


        vis = colorize_draw_agent_and_fit_to_height(td, canvas_h)  # BGR
        H_map = td["map"].shape[0]
        scale = float(vis.shape[0]) / float(H_map)

        # id->idx（节点ID可能不连续，避免直接用id当索引）
        id2idx = {n.id: i for i, n in enumerate(self.graph.nodes)}

        # 计算每个节点的原始像素坐标
        node_px = []
        for n in self.graph.nodes:
            col, row = self.world_xz_to_rot_px(env, n.center[0], n.center[2], td)
            node_px.append((col, row))
        return vis, td, scale, node_px, id2idx

    # ================= NEW(1): 只画 Graph（节点/边） =================
    def visualize_graph_only(self,
                             env,
                             out_path: str,
                             show_nodes: bool = True,
                             show_edges: bool = True,
                             node_radius: int = 2,
                             edge_thickness: int = 1,
                             canvas_h: int = 1024):
        vis, td, scale, node_px, id2idx = self._prepare_canvas_and_nodes(env, canvas_h)

        def up(p): return (p[0] * scale, p[1] * scale)

        if show_nodes:
            for p in node_px:
                self._draw_dot(vis, up(p), color=(180, 220, 255), r=node_radius)

        if show_edges:
            # adj 的键是 node_id，不一定连续；用 id2idx 做映射
            for uid, nbrs in self.graph.adj.items():
                ui = id2idx.get(uid, None)
                if ui is None:
                    continue
                pu = up(node_px[ui])
                for vid, _dist in nbrs:
                    if vid <= uid:
                        continue  # 无向边画一次
                    vi = id2idx.get(vid, None)
                    if vi is None:
                        continue
                    pv = up(node_px[vi])
                    cv2.line(vis, (int(pu[0]), int(pu[1])), (int(pv[0]), int(pv[1])),
                             color=(80, 160, 80), thickness=edge_thickness, lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[viz-graph] saved -> {out_path}")
        return out_path

    # ================= NEW(2): 叠加 start/goal 的 top-K（不做本地化） =================
    def visualize_localization_results_on_graph(self,
                                                env,
                                                episode_result: dict,
                                                out_path: str,
                                                topk: int = 5,
                                                show_nodes: bool = False,   # 默认不画节点
                                                show_edges: bool = False,   # 默认不画边
                                                node_radius: int = 14,
                                                canvas_h: int = 1024):
        """
        仅在 topdown 上叠加：
        - start_node / goal_node
        - start_topk / goal_topk
        - （可选）measurement 中的真值 agent/goal 像素（如果存在）
        不绘制图中的其他节点和边。
        episode_result 形如：
        {
        "episode_index": 0,
        "start_node": 200,
        "goal_node": 189,
        "start_top5": [[200,0.75],[199,0.72],...],
        "goal_top5":  [[189,0.47],[190,0.44],...]
        }
        """
        # 仍然复用底图构建与世界->像素映射，但不绘制全图节点和边
        vis, td, scale, node_px, id2idx = self._prepare_canvas_and_nodes(env, canvas_h)
        def up(p): return (p[0] * scale, p[1] * scale)

        # 读取本地化结果
        s_top = episode_result.get("start_top5", [])
        g_top = episode_result.get("goal_top5", [])

        def _norm_list(L):
            out = []
            for it in L:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    out.append((int(it[0]), float(it[1])))
                elif isinstance(it, dict) and "id" in it and "score" in it:
                    out.append((int(it["id"]), float(it["score"])))
            return out

        s_top = _norm_list(s_top)[:topk]
        g_top = _norm_list(g_top)[:topk]

        # ---------- 1) 画真值起点/终点（优先 xyz；否则回退 node id） ----------
        sx, sy, sz = [float(v) for v in episode_result["start_xyz"]]
        gx, gy, gz = [float(v) for v in episode_result["goal_xyz"]]
        ps = up(self.world_xz_to_rot_px(env, sx, sz, td))
        pg = up(self.world_xz_to_rot_px(env, gx, gz, td))
        # 绿色：起点真值；红色：目标真值
        self._draw_dot(vis, ps, color=(0, 255, 0), r=node_radius)
        self._draw_text(vis, ps, "START_TRUE", (0, 255, 0))
        self._draw_dot(vis, pg, color=(0, 0, 255), r=node_radius)
        self._draw_text(vis, pg, "GOAL_TRUE", (0, 0, 255))

        # 仅绘制 start/goal 的 top-k
        # 颜色规范：start-topk 青色(0,200,255)；goal-topk 橙色(255,130,0)
        for rank, (nid, sim) in enumerate(s_top, start=1):
            idx = id2idx.get(nid, None)
            if idx is None: continue
            p = up(node_px[idx])
            self._draw_dot(vis, p, color=(0, 200, 255), r=node_radius)
            self._draw_text(vis, p, f"S{rank}:{nid}({sim:.2f})", color=(0, 200, 255))

        for rank, (nid, sim) in enumerate(g_top, start=1):
            idx = id2idx.get(nid, None)
            if idx is None: continue
            p = up(node_px[idx])
            self._draw_dot(vis, p, color=(255, 130, 0), r=node_radius)
            self._draw_text(vis, p, f"G{rank}:{nid}({sim:.2f})", color=(255, 130, 0))

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[viz-loc] saved -> {out_path}")
        return out_path

    def visualize_localization_results_on_graph_pdf(
        self,
        env,
        episode_result: dict,
        out_path: str,
        topk: int = 5,
        show_nodes: bool = False,
        show_edges: bool = False,
        node_radius: int = 14,
        canvas_h: int = 1024,
    ):
        """
        生成 PDF：背景是 topdown bitmap，节点/文字是矢量，方便 AI 编辑。
        """
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        # 1. 仍然复用你自己的准备逻辑
        vis, td, scale, node_px, id2idx = self._prepare_canvas_and_nodes(env, canvas_h)
        def up(p): return (p[0] * scale, p[1] * scale)

        # 2. 解析 top-k 列表
        s_top = episode_result.get("start_top5", [])
        g_top = episode_result.get("goal_top5", [])

        def _norm_list(L):
            out = []
            for it in L:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    out.append((int(it[0]), float(it[1])))
                elif isinstance(it, dict) and "id" in it and "score" in it:
                    out.append((int(it["id"]), float(it["score"])))
            return out

        s_top = _norm_list(s_top)[:topk]
        g_top = _norm_list(g_top)[:topk]

        # 3. 计算真值起点/终点的像素坐标
        sx, sy, sz = [float(v) for v in episode_result["start_xyz"]]
        gx, gy, gz = [float(v) for v in episode_result["goal_xyz"]]
        ps = up(self.world_xz_to_rot_px(env, sx, sz, td))
        pg = up(self.world_xz_to_rot_px(env, gx, gz, td))

        # 4. 用 matplotlib 画：背景 bitmap + 矢量 overlay
        #    注意：td 是未经 scale 的 topdown（按你 prepare 里的定义来）
        h, w = vis.shape[:2]
        fig_w = w * scale / 100.0   # 调整一下比例，让分辨率别太怪
        fig_h = h * scale / 100.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

        # 背景仍然是位图
        ax.imshow(vis, origin="upper")

        # 起点/终点真值（绿色/红色）
        ax.scatter(ps[0], ps[1], s=node_radius**2, edgecolors="none")
        ax.text(ps[0], ps[1], "START_TRUE", color="green",
                fontsize=10, ha="center", va="bottom")
        ax.scatter(pg[0], pg[1], s=node_radius**2, edgecolors="none")
        ax.text(pg[0], pg[1], "GOAL_TRUE", color="blue",
                fontsize=10, ha="center", va="bottom")

        # start top-k：青色；goal top-k：橙色
        # for rank, (nid, sim) in enumerate(s_top, start=1):
        #     idx = id2idx.get(nid, None)
        #     if idx is None:
        #         continue
        #     p = up(node_px[idx])
        #     ax.scatter(p[0], p[1], s=node_radius**2, edgecolors="none")
        #     ax.text(p[0], p[1], f"S{rank}:{nid}({sim:.2f})",
        #             fontsize=8, ha="center", va="bottom")

        for rank, (nid, sim) in enumerate(g_top, start=1):
            idx = id2idx.get(nid, None)
            if idx is None:
                continue
            p = up(node_px[idx])
            ax.scatter(p[0], p[1], s=node_radius**2, edgecolors="none")
            ax.text(p[0], p[1], f"G{rank}:{nid}({sim:.2f})",
                    fontsize=8, ha="center", va="bottom")

        ax.axis("off")

        # 5. 保存为 PDF
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        print(f"[viz-loc] saved (vector overlays) -> {pdf_path}")
        return pdf_path

    # ================= NEW(3): 叠加节点路径（不做规划） =================
    def visualize_path_on_graph(
        self,
        env,
        node_path: List[int],
        out_path: str,
        show_nodes: bool = True,
        show_edges: bool = True,
        node_radius: int = 2,
        path_thickness: int = 3,
        canvas_h: int = 1024,
        # 新增↓↓↓
        true_start_xyz=None,                 # (x,y,z) or None
        true_goal_xyz=None,                  # (x,y,z) or None
        prefix_segment=None,                 # ((sx,sz),(nx,nz)) or None
        suffix_segment=None,                 # ((gx0,gz0),(gx1,gz1)) or None
    ):
        vis, td, scale, node_px, id2idx = self._prepare_canvas_and_nodes(env, canvas_h)
        def up(p): return (p[0] * scale, p[1] * scale)

        # 底图：节点/边
        if show_nodes:
            for p in node_px:
                self._draw_dot(vis, up(p), color=(180, 220, 255), r=node_radius)
        if show_edges:
            for uid, nbrs in self.graph.adj.items():
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
            self._draw_dot(vis, a0, color=(0, 255, 255), r=7)
            self._draw_text(vis, a0, f"PATH_START:{node_path[0]}", (0, 255, 255))
            self._draw_dot(vis, an, color=(255, 255, 0), r=7)
            self._draw_text(vis, an, f"PATH_GOAL:{node_path[-1]}", (255, 255, 0))

        # —— 新增：画真实起点 & 前缀直连（虚线/青色系）
        if true_start_xyz is not None:
            # 世界→像素（复用 _world_to_map_px）
            sx_px, sy_px = self.world_xz_to_rot_px(env, true_start_xyz[0], true_start_xyz[2], td)
            S = (int(sx_px*scale), int(sy_px*scale))
            self._draw_dot(vis, S, color=(0, 200, 255), r=8)
            self._draw_text(vis, S, "TRUE_START", (0, 200, 255))
        if prefix_segment is not None:
            (sx, sz), (nx, nz) = prefix_segment
            s_px = self.world_xz_to_rot_px(env, sx, sz, td); s_px = (int(s_px[0]*scale), int(s_px[1]*scale))
            n_px = self.world_xz_to_rot_px(env, nx, nz, td); n_px = (int(n_px[0]*scale), int(n_px[1]*scale))
            # 画虚线
            cv2.line(vis, s_px, n_px, color=(0, 200, 255), thickness=path_thickness, lineType=cv2.LINE_AA)
            # 虚线效果：在上面叠一层小圆点
            for t in np.linspace(0, 1, 15):
                x = int(s_px[0]*(1-t) + n_px[0]*t)
                y = int(s_px[1]*(1-t) + n_px[1]*t)
                cv2.circle(vis, (x, y), 2, (0, 200, 255), -1, lineType=cv2.LINE_AA)

        # —— 新增：画真实终点 & 后缀直连（虚线/橙色系）
        if true_goal_xyz is not None:
            gx_px, gy_px = self.world_xz_to_rot_px(env, true_goal_xyz[0], true_goal_xyz[2], td)
            G = (int(gx_px*scale), int(gy_px*scale))
            self._draw_dot(vis, G, color=(255, 130, 0), r=8)
            self._draw_text(vis, G, "TRUE_GOAL", (255, 130, 0))
        if suffix_segment is not None:
            (x0, z0), (x1, z1) = suffix_segment
            a_px = self.world_xz_to_rot_px(env, x0, z0, td); a_px = (int(a_px[0]*scale), int(a_px[1]*scale))
            g_px = self.world_xz_to_rot_px(env, x1, z1, td); g_px = (int(g_px[0]*scale), int(g_px[1]*scale))
            cv2.line(vis, a_px, g_px, color=(255, 130, 0), thickness=path_thickness, lineType=cv2.LINE_AA)
            for t in np.linspace(0, 1, 15):
                x = int(a_px[0]*(1-t) + g_px[0]*t)
                y = int(a_px[1]*(1-t) + g_px[1]*t)
                cv2.circle(vis, (x, y), 2, (255, 130, 0), -1, lineType=cv2.LINE_AA)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, vis)
        print(f"[viz-path] saved -> {out_path}")
        return out_path


    ###############################
    # 走到matched goal后进行再次匹配
    ###############################
    # ========= 工具 =========

    # ========= global feature match =========
    @staticmethod
    def _nearest_multiple(deg: float, base: float):
        """
        把期望角度deg对齐为base(=TURN_ANGLE)的整数倍
        返回: (实际角度deg_aligned, 每次需要的turn动作次数n_turns)
        """
        m = max(1, int(round(float(deg) / float(base))))
        return float(m * base), int(m)

    @staticmethod
    def _turn(env, direction: str, n_steps: int):
        act = "turn_left" if direction == "left" else "turn_right"
        for _ in range(int(n_steps)):
            env.step({"action": act})

    @torch.inference_mode()
    def _score_global_list(self, goal_rgb, obs_rgbs):
        """
        robot.encoder.encode(rgb) -> np.float32[D]（已L2 norm）
        返回与goal的余弦分数列表，与obs_rgbs等长
        """
        g = self.encoder.encode(goal_rgb)  # [D], L2-normalized
        scores = []
        for img in obs_rgbs:
            f = self.encoder.encode(img)   # [D], L2-normalized
            scores.append(float(np.dot(g, f)))
        return scores

    # ======= token feature match =========
    @torch.inference_mode()
    def _tokens_from_rgb(self, rgb) -> torch.Tensor:
        """
        返回归一化的 tokens: [N, D] (float32, CUDA/CPU 取决于模型所在设备)
        """
        toks, _ = self.encoder.extract_patch_tokens(rgb)  # np.float32 [Hp, Wp, D]
        t = torch.from_numpy(toks).reshape(-1, toks.shape[-1])       # [N, D]
        t = t.to(next(self.encoder.model.parameters()).device, non_blocking=True)
        t = F.normalize(t, dim=-1)
        return t

    @torch.inference_mode()
    def _score_patch_pair(self, goal_tokens: torch.Tensor,
                        obs_tokens: torch.Tensor,
                        method: str = "chamfer",
                        temperature: float = 0.07,
                        topk: int = 0) -> float:
        """
        goal_tokens, obs_tokens: [N, D] 已 L2 归一化
        返回一个标量分数（越大越相似）
        """
        # 余弦相似度矩阵 S: [Ng, No]
        S = goal_tokens @ obs_tokens.t()  # cos 因为已 L2 norm

        if topk and method in ("chamfer", "softmax"):
            # 可选降噪：仅保留每行/每列 top-k（稀疏化），减少背景干扰
            # 这里先做行向 topk，列向可选
            vals, idx = torch.topk(S, k=min(topk, S.shape[1]), dim=1)
            mask = torch.zeros_like(S)
            mask.scatter_(1, idx, 1.0)
            S = S * mask + (1.0 - mask) * (-1e4)   # 屏蔽非topk

        if method == "mean_pool":
            # 等效于全局平均池化后做余弦
            g = F.normalize(goal_tokens.mean(dim=0, keepdim=True), dim=-1)  # [1,D]
            o = F.normalize(obs_tokens.mean(dim=0,  keepdim=True), dim=-1)
            score = float((g @ o.t()).item())
            return score

        elif method == "chamfer":
            # goal→obs：每个 goal token 取 obs 中最大相似度，取均值
            g2o = S.max(dim=1).values.mean()
            # obs→goal：每个 obs token 取 goal 中最大相似度，取均值
            o2g = S.max(dim=0).values.mean()
            score = float(0.5 * (g2o + o2g).item())
            return score

        elif method == "softmax":
            # goal→obs：soft 对齐
            g2o = (F.softmax(S / max(1e-6, temperature), dim=1) * S).sum(dim=1).mean()
            # obs→goal：soft 对齐（转置）
            o2g = (F.softmax(S.t() / max(1e-6, temperature), dim=1) * S.t()).sum(dim=1).mean()
            score = float(0.5 * (g2o + o2g).item())
            return score

        else:
            raise ValueError(f"unknown method: {method}")

    @torch.inference_mode()
    def _score_patch_list(self, goal_rgb, obs_rgbs,
                        method: str = "chamfer",
                        temperature: float = 0.07,
                        topk: int = 0) -> list:
        """
        用 patch tokens 计算 goal 对 obs 列表的分数
        """
        g_tok = self._tokens_from_rgb(goal_rgb)  # [Ng, D]
        scores = []
        for img in obs_rgbs:
            o_tok = self._tokens_from_rgb(img)  # [No, D]
            scores.append(self._score_patch_pair(g_tok, o_tok, method=method,
                                            temperature=temperature, topk=topk))
        return scores

    def align_to_goal_coarse_dinov2(self,
        env,
        goal_rgb,                # (H,W,3) 或等价RGB
        step_deg: float = 30.0,  # 粗扫步长（会与TURN_ANGLE对齐）
        prefer_left: bool = True,
        verbose: bool = True,
    ):
        """
        只做粗筛（DINOv2 global feature 余弦），转一圈采样，选最佳角度并对齐。
        返回:
        {
        "step_deg": 对齐后的步长(度),
        "scores":   每个朝向的分数列表,
        "best_idx": 最优朝向的索引(0..steps_per_round-1),
        "best_angle_deg": 最优相对角度（相对起始朝向，左转为正）
        }
        """
        # 1) 拿到TURN_ANGLE并与之对齐
        try:
            turn_angle = self.args.turn_left
        except Exception:
            turn_angle = 10.0  # 兜底
        step_deg_aligned, n_turns = self._nearest_multiple(step_deg, turn_angle)

        # 2) 计算一圈采样次数，并转一圈采样
        steps_per_round = int(round(360.0 / step_deg_aligned))
        if steps_per_round <= 0:
            steps_per_round = 1

        obs_rgbs = []
        # 设计为：先采样再转动，这样采到的是当前朝向
        for _ in range(steps_per_round):
            obs = env.sim.get_sensor_observations(0)
            obs_rgbs.append(obs["rgb"])
            self._turn(env, "left" if prefer_left else "right", n_turns)

        # 经过steps_per_round次等步长旋转，朝向应回到起点（对齐后保证整数整圈）

        # 3) 用DINOv2全局特征与goal算余弦分数
        #scores = self._score_global_list(goal_rgb, obs_rgbs)
        scores = self._score_patch_list(goal_rgb, obs_rgbs)
        best_idx = int(np.argmax(scores))

        # 4) 从“起始朝向”转到best_idx对应的朝向：选最少步数方向
        left_steps  = best_idx
        right_steps = steps_per_round - best_idx
        real_turn_steps = 0
        real_turn_direction = "left"
        if right_steps < left_steps:
            real_turn_steps = right_steps
            real_turn_direction = "right"
            # self._turn(env, "right", right_steps * n_turns)
            final_angle_deg = - right_steps * step_deg_aligned
        else:
            real_turn_steps = left_steps
            real_turn_direction = "left"
            # self._turn(env, "left",  left_steps  * n_turns)
            final_angle_deg = + left_steps  * step_deg_aligned

        if verbose:
            print(f"[CoarseAlign] step={step_deg_aligned}°, samples={steps_per_round}, "
                f"best_idx={best_idx}, final≈{final_angle_deg:.1f}° "
                f"(score={scores[best_idx]:.4f})")

        return {
            "step_deg": float(step_deg_aligned),
            "scores":   [float(s) for s in scores],
            "best_idx": best_idx,
            "best_angle_deg": float(final_angle_deg),
            "steps_per_round": int(steps_per_round),
            "real_turn": int(real_turn_steps),
            "real_turn_direction": real_turn_direction
        }
    
    def align_to_goal_coarse_dinov2_real_robot(self,
        obs,
        goal_rgb,                # (H,W,3) 或等价RGB
        step_deg: float = 30.0,  # 粗扫步长（会与TURN_ANGLE对齐）
        prefer_left: bool = True,
        verbose: bool = True,
    ):
        """
        只做粗筛（DINOv2 global feature 余弦），转一圈采样，选最佳角度并对齐。
        返回:
        {
        "step_deg": 对齐后的步长(度),
        "scores":   每个朝向的分数列表,
        "best_idx": 最优朝向的索引(0..steps_per_round-1),
        "best_angle_deg": 最优相对角度（相对起始朝向，左转为正）
        }
        """
        # 1) 拿到TURN_ANGLE并与之对齐
        try:
            turn_angle = self.args.turn_left
        except Exception:
            turn_angle = 10.0  # 兜底
        step_deg_aligned, n_turns = self._nearest_multiple(step_deg, turn_angle)

        # 2) 计算一圈采样次数，并转一圈采样
        steps_per_round = int(round(360.0 / step_deg_aligned))
        if steps_per_round <= 0:
            steps_per_round = 1

        obs_rgbs = []
        # 设计为：先采样再转动，这样采到的是当前朝向
        for _ in range(steps_per_round):
            obs = env.sim.get_sensor_observations(0)
            obs_rgbs.append(obs["rgb"])
            self._turn(env, "left" if prefer_left else "right", n_turns)

        # 经过steps_per_round次等步长旋转，朝向应回到起点（对齐后保证整数整圈）

        # 3) 用DINOv2全局特征与goal算余弦分数
        #scores = self._score_global_list(goal_rgb, obs_rgbs)
        scores = self._score_patch_list(goal_rgb, obs_rgbs)
        best_idx = int(np.argmax(scores))

        # 4) 从“起始朝向”转到best_idx对应的朝向：选最少步数方向
        left_steps  = best_idx
        right_steps = steps_per_round - best_idx
        real_turn_steps = 0
        real_turn_direction = "left"
        if right_steps < left_steps:
            real_turn_steps = right_steps
            real_turn_direction = "right"
            # self._turn(env, "right", right_steps * n_turns)
            final_angle_deg = - right_steps * step_deg_aligned
        else:
            real_turn_steps = left_steps
            real_turn_direction = "left"
            # self._turn(env, "left",  left_steps  * n_turns)
            final_angle_deg = + left_steps  * step_deg_aligned

        if verbose:
            print(f"[CoarseAlign] step={step_deg_aligned}°, samples={steps_per_round}, "
                f"best_idx={best_idx}, final≈{final_angle_deg:.1f}° "
                f"(score={scores[best_idx]:.4f})")

        return {
            "step_deg": float(step_deg_aligned),
            "scores":   [float(s) for s in scores],
            "best_idx": best_idx,
            "best_angle_deg": float(final_angle_deg),
            "steps_per_round": int(steps_per_round),
            "real_turn": int(real_turn_steps),
            "real_turn_direction": real_turn_direction
        }
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dataset", type=str, default='hm3d')

    parser.add_argument("--graph_json", type=str, required=True)
    parser.add_argument("--explore_npz", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dino_arch", type=str, default="dinov2_vitl14")
    parser.add_argument("--use_env", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--nav_task", type=str, default='objnav') # objnav, imgnav, ovon, r2r
    
    HABITAT_ROOT_DIR = "/home/wangbo/codes/BSC-Nav/third-party/habitat-lab"
    parser.add_argument("--HM3D_CONFIG_PATH", type=str, default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
    parser.add_argument("--MP3D_CONFIG_PATH", type=str, default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")
    
    parser.add_argument("--HM3D_SCENE_PREFIX", type=str, default="/nas_dataset/wangbo/HM3D")
    parser.add_argument("--HM3D_EPISODE_PREFIX", type=str, default="/home/wangbo/codes/BSC-Nav/data_episode/objnav/objectnav_hm3d_v2/val_mini/val_mini.json.gz")

    parser.add_argument("--MP3D_SCENE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/")
    parser.add_argument("--MP3D_EPISODE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/baselines/Pixel-Navigator/checkpoints/objectnav_mp3d_v1/val/val.json.gz")

    parser.add_argument("--image_hfov", type=int, default=90)

    
    parser.add_argument("--eval_episodes", type=int, default=1000)    # hm3d-0.1 2000 / mp3d-0.1 2195  
    parser.add_argument("--max_episode_steps", type=int, default=5000)  
    parser.add_argument("--success_distance", type=float, default=1.0)  
    args = parser.parse_args()

    robot = ImageNavGraphRobot(args.graph_json, args.explore_npz,
                               dino_arch=args.dino_arch, device=args.device)
    if not args.use_env:
        print("Ready. 你可以在外部脚本中：robot.localize_start_goal_from_obs(obs) / robot.plan_waypoints(...)。")
    else:
        # 仅 Demo：从你的 ImageNav 环境拿 obs 做一次本地化+规划
        try:
            from env import get_objnav_env
        except Exception as e:
            raise RuntimeError("找不到 env.get_objnav_env；请去掉 --use_env 或提供 ImageNav 环境。") from e

        env = get_objnav_env(args)
        for ep in range(args.episodes):
            obs = env.reset()
            s_id, g_id, s_top, g_top = robot.localize_start_goal_from_obs(obs)
            path = robot.plan_waypoints(s_id, g_id)
            print(f"[Episode {ep}] start->goal = {s_id} -> {g_id}")
            print(f"  start@top5: {s_top}")
            print(f"  goal@top5 : {g_top}")
            print(f"  node path : {path}")
