### Environment & Dataset preparation

1. Preparing **Habitat-lab** & **Habitat-sim** Env (refer https://github.com/facebookresearch/habitat-lab for detailed installation instructions)

```
# create conda env
conda create -n mgnav python=3.9 cmake=3.14.0
conda activate mgnav

# install habitat-sim with bullet physics 
# download habitat-sim-0.3.3-py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24.conda from https://anaconda.org/channels/aihabitat/packages/habitat-sim/files?name=headles

conda install habitat-sim-0.3.3-py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24.conda

pip install -r requirements.txt 
# You may need to manually resolve version conflicts

# install habitat-lab stable version
cd third-party/habitat-lab
pip install -e habitat-lab  # install habitat_lab
pip install -e habitat-baselines 
```

2. Download scene dataset and benchmark episode files

- BSC-Nav is evaluated under HM3D and MP3D scene datasets, for download, We strongly recommend referring to the following guidelines:

  HM3D: http://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d

  MP3D: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset

- The episode file is used for benchmarking and can be downloaded from the following address:

*Image-instance navigation*: https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md, we need `instance_imagenav_hm3d_v3.zip`

- make sure it is unzip and organized as follows:

  ```
  -- MG-Nav
     -- data_episode
        -- imagenav
              ...
  ```

3. Install Dinov2 &  Grounded-Sam-2

   cd third-party

- Dinov2: git clone https://github.com/facebookresearch/dinov2.git
- Grounded-Sam-2: git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
- install Grounded-sam-2 following its instruction
- move grounded_sam2_wrapper.py to third-party/Grounded-SAM-2
- download grounding-ding-tiny from https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/main to third-party/Grounded-SAM-2

4. Install NavDP (https://github.com/InternRobotics/NavDP)

   Create conda environment and install the dependency

   ```
   cd third-party/NavDP/baselines/navdp/
   conda create -n navdp python=3.10
   conda activate navdp
   pip install -r requirements.txt
   ```

5. Run MG-Nav 

   Run the following line to start exploration and graph construction: 

   ```
   # You can choose the specific scene in construct_graph_total.py, e.g. "00810-CrMo8WxCyVb": "CrMo8WxCyVb",
   
   # Explore the environment
   CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py --explore_map --semantic_analyze
   
   # Construct the graph of different floors under different graph sizes (choose the graph sizes as you need)
   
   CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py --construct_graph --visualize_graph --floor_idx 0 --min_dis 1.0 --radius 0.5
   
   CUDA_VISIBLE_DEVICES=0 python construct_graph_total.py --construct_graph --visualize_graph --floor_idx 1 --min_dis 1.5 --radius 0.8
   ```

   Run the following line to start navdp server: 

   ```
   conda activate navdp
   cd third-party/NavDP
   
   CUDA_VISIBLE_DEVICES=0 python baselines/navdp/navdp_server_geometry.py --port 6666 --checkpoint ./checkpoints/checkpoint-43956navdp-onlyproj.ckpt
   
   ```

   After start the navdp server, a new window should be created, And then run the following line to start navigation

```
conda activate mgnav

# choose the same scene in run_navdp_follow_path_continuous_total.py to navigation

# slow running with visualization video
CUDA_VISIBLE_DEVICES=0 python run_navdp_follow_path_continuous_total.py --rpc_port 6666 --max_total_steps 500 --eval_episodes 1000 --floor_idx 0 --min_dis 1.0 --radius 0.5

# fast running without visualizaiton video
CUDA_VISIBLE_DEVICES=0 python run_navdp_follow_path_continuous_total_quick.py --rpc_port 6666 --max_total_steps 500 --eval_episodes 1000 --floor_idx 0 --min_dis 1.0 --radius 0.5

```



