# Conda 环境 `mgnav`（MG-Nav 主环境）

## 1. 创建环境

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
conda create -n mgnav python=3.9 cmake=3.14.0 -y
conda activate mgnav
```

## 2. Habitat-Sim 0.3.3

从 [aihabitat/habitat-sim](https://anaconda.org/aihabitat/habitat-sim/files?version=0.3.3) 下载与 **py3.9 headless bullet linux** 匹配的 `.conda`，例如：

```bash
# 需 conda-forge 提供 python_abi；并先 unset 代理
conda install -c conda-forge -c aihabitat -y \
  "habitat-sim=0.3.3=py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24"
```

## 3. Python 依赖与 Habitat-Lab

```bash
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav
pip install -r requirements.txt
# 若冲突，保留 numpy==1.26.4

cd third-party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

## 4. 第三方仓库

```bash
cd third-party
git clone https://github.com/facebookresearch/dinov2.git
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cp ../grounded_sam2_wrapper.py Grounded-SAM-2/
```

权重（示例）：

- SAM2：`Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt`
- Grounding DINO tiny：`Grounded-SAM-2/grounding-dino-tiny/`（HuggingFace IDEA-Research/grounding-dino-tiny）

按 Grounded-SAM-2 官方 README 安装其 Python 依赖。

## 5. HM3D 语义配置

将 Matterport **`hm3d-val-semantic-configs-v0.2.tar`** 解压到：

`~/data/processed/unigoal_datasets/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json`

## 6. 数据软链（仓库内）

```bash
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav
ln -sfn ~/data/processed/unigoal_datasets/instance_imagenav/hm3d/v3 \
  data_episode/imagenav/instance_imagenav_hm3d_v3
mkdir -p data
ln -sfn ~/data/processed/unigoal_datasets/scene_datasets/hm3d_v0.2 data/hm3d
```

## 7. 验证

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```
