import habitat_sim
# @title Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import math
import os
import random

import git
import imageio
import magnum as mn
import numpy as np

from habitat_sim.gfx import LightInfo, LightPositionModel
import magnum as mn

from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut


output_path = "/home/wangbo/codes/NavDP/MP3d_output"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# def make_cfg(settings):
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.gpu_device_id = 0
#     sim_cfg.scene_id = settings["scene"]
#     sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
#     sim_cfg.enable_physics = settings["enable_physics"]

#     # Note: all sensors must have the same resolution
#     sensor_specs = []

#     color_sensor_spec = habitat_sim.sensor.CameraSensorSpec()
#     color_sensor_spec.uuid = "color_sensor"
#     color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
#     color_sensor_spec.resolution = [settings["height"], settings["width"]]
#     color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
#     color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     sensor_specs.append(color_sensor_spec)

#     depth_sensor_spec = habitat_sim.CameraSensorSpec()
#     depth_sensor_spec.uuid = "depth_sensor"
#     depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
#     depth_sensor_spec.resolution = [settings["height"], settings["width"]]
#     depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
#     depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     sensor_specs.append(depth_sensor_spec)

#     # semantic_sensor_spec = habitat_sim.CameraSensorSpec()
#     # semantic_sensor_spec.uuid = "semantic_sensor"
#     # semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
#     # semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
#     # semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
#     # semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     # sensor_specs.append(semantic_sensor_spec)

#     # Here you can specify the amount of displacement in a forward action and the turn angle
#     agent_cfg = habitat_sim.agent.AgentConfiguration()
#     agent_cfg.sensor_specifications = sensor_specs
#     agent_cfg.action_space = {
#         "stop": habitat_sim.agent.ActionSpec("stop"),
#         "move_forward": habitat_sim.agent.ActionSpec(
#             "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
#         ),
#         "turn_left": habitat_sim.agent.ActionSpec(
#             "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
#         ),
#         "turn_right": habitat_sim.agent.ActionSpec(
#             "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
#         ),
        
#     }

#     return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_cfg(settings, turn_degree=30.0, forward_step=0.25,
             cam_near=0.01, cam_far=20.0):   # ← 新增：近/远裁剪面
    # ---- Simulator config ----
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # ---- Sensor specs ----
    sensor_specs = []
    try:
        from habitat_sim.sensor import SensorSpec
    except Exception:
        from habitat_sim._ext.habitat_sim_bindings import SensorSpec

    def _set_clip_planes(spec, near_val, far_val):
        # 兼容不同版本的属性命名
        for k in ("near", "clip_near"):
            if hasattr(spec, k):
                setattr(spec, k, float(near_val))
        for k in ("far", "clip_far"):
            if hasattr(spec, k):
                setattr(spec, k, float(far_val))

    def build_sensor(uuid: str, sensor_type):
        spec = SensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [settings["height"], settings["width"]]  # [H, W]
        spec.position = [0.0, settings["sensor_height"], 0.0]
        _set_clip_planes(spec, cam_near, cam_far)  # ← 设置近/远裁剪面
        return spec

    # RGB
    color_sensor_spec = build_sensor("color_sensor", habitat_sim.SensorType.COLOR)
    sensor_specs.append(color_sensor_spec)

    # 深度
    depth_sensor_spec = build_sensor("depth_sensor", habitat_sim.SensorType.DEPTH)
    sensor_specs.append(depth_sensor_spec)

    # ---- Agent config ----
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "stop": habitat_sim.agent.ActionSpec("stop", habitat_sim.agent.ActuationSpec(amount=0.0)),
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=forward_step)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=turn_degree)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=turn_degree)),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None,prefix="topdown"):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # 保存图像
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f"{prefix}_map.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()