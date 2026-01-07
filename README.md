# MG-Nav

MG-Nav: Dual-Scale Visual Navigation via Sparse Spatial Memory
'https://arxiv.org/abs/2511.22609'

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Third-Party Dependencies](#third-party-dependencies)
- [NavDP Setup](#navdp-setup)
- [Run MG-Nav](#run-mg-nav)

---

## Environment Setup

### 1) Create Conda Environment

```bash
conda create -n mgnav python=3.9 cmake=3.14.0
conda activate mgnav
```

### 2) Install Habitat-Sim (headless + bullet)

Download precompiled Habitat-Sim package (example: Linux, py3.9, headless, bullet):
- https://anaconda.org/aihabitat/habitat-sim/files?version=0.3.3

```bash
# Example:
# habitat-sim-0.3.3-py3.9_headless_bullet_linux_*.conda
conda install habitat-sim-0.3.3-py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24.conda
```

### 3) Install Python Dependencies

```bash
pip install -r requirements.txt
# You may need to manually resolve version conflicts
```

### 4) Install Habitat-Lab

```bash
cd third-party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

---

## Dataset Preparation

MG-Nav is evaluated on **HM3D** and **MP3D**.

### 1) Download Scene Datasets

- HM3D: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d
- MP3D: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset

### 2) Download Episode Files (for benchmarking)

For *Image / Instance Navigation*:
- https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md  
  We need: `instance_imagenav_hm3d_v3.zip`

Unzip and organize as:

```text
MG-Nav/
└── data_episode/
    └── imagenav/
        └── ...
```

---

## Third-Party Dependencies

All third-party repos are placed under `third-party/`.

### 1) Install DINOv2

```bash
cd third-party
git clone https://github.com/facebookresearch/dinov2.git
```

### 2) Install Grounded-SAM-2

```bash
cd third-party
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```

Then:
1. Install Grounded-SAM-2 following its official instructions.
2. Move `grounded_sam2_wrapper.py` to:
   ```text
   third-party/Grounded-SAM-2/
   ```
3. Download `grounding-dino-tiny` from:
   - https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main  
   and place it under:
   ```text
   third-party/Grounded-SAM-2/
   ```

---

### 3) NavDP Setup

NavDP runs in a separate conda environment.

```bash
cd third-party/NavDP/baselines/navdp

conda create -n navdp python=3.10
conda activate navdp
pip install -r requirements.txt
```

Please download the pretrained NavDP checkpoint (`.ckpt`) from the https://drive.google.com/file/d/1m3dr3PKgKRADErC61y2aTneMOYWozljU/view?usp=drive_link and put it in `third-party/NavDP/checkpoints/`.

---

## Run MG-Nav

### 1) Exploration & Graph Construction

Select the target scene in `construct_graph_total.py`, e.g. "00810-CrMo8WxCyVb": "CrMo8WxCyVb"

Explore the environment:

```bash
CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py --explore_map --semantic_analyze
```

Construct graphs (different floors / graph sizes):

```bash
# Floor 0 & small node
CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py \
  --construct_graph --visualize_graph \
  --floor_idx 0 --min_dis 1.0 --radius 0.5

# Floor 1 & large node
CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py \
  --construct_graph --visualize_graph \
  --floor_idx 1 --min_dis 1.5 --radius 0.8
```

### 2) Start NavDP Server

```bash
conda activate navdp
cd third-party/NavDP

CUDA_VISIBLE_DEVICES=0 python baselines/navdp/navdp_server_geometry.py \
  --port 6666 \ # you can set random ports as you need
  --checkpoint ./checkpoints/checkpoint-43956navdp-onlyproj.ckpt
```

### 3) Run Navigation

```bash
conda activate mgnav
```

Slow running (with visualization video):

```bash
CUDA_VISIBLE_DEVICES=0 python run_navdp_follow_path_continuous_total.py \
  --rpc_port 6666 --max_total_steps 500 --eval_episodes 1000 \
  --floor_idx 0 --min_dis 1.0 --radius 0.5
```

Fast running (without visualization video):

```bash
CUDA_VISIBLE_DEVICES=0 python run_navdp_follow_path_continuous_total_quick.py \
  --rpc_port 6666 --max_total_steps 500 --eval_episodes 1000 \
  --floor_idx 0 --min_dis 1.0 --radius 0.5
```




