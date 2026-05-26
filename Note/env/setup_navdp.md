# Conda 环境 `navdp`（NavDP RPC 服务）

## 1. 创建并安装

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
conda create -n navdp python=3.10 -y
conda activate navdp
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav/third-party/NavDP/baselines/navdp
pip install -r requirements.txt
```

## 2. 下载 checkpoint

README 链接：  
https://drive.google.com/file/d/1m3dr3PKgKRADErC61y2aTneMOYWozljU/view

放置为：

`third-party/NavDP/checkpoints/checkpoint-43956navdp-onlyproj.ckpt`

## 3. 启动服务

```bash
conda activate navdp
cd /home/ial-chenzm/workspace/10_baselines/MG-CAM/MG-Nav/third-party/NavDP
CUDA_VISIBLE_DEVICES=3 python baselines/navdp/navdp_server_geometry.py \
  --port 6666 \
  --checkpoint ./checkpoints/checkpoint-43956navdp-onlyproj.ckpt
```

## 4. 与主环境联调

另开终端 `conda activate mgnav`，运行 `Note/results/launch_mg_nav.py`（需已有 `memory/` 图）。
