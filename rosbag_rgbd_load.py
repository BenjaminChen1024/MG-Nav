# rgbd_odom_source.py
import rclpy
from rclpy.node import Node
from collections import deque
import numpy as np
import cv2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def rgb_to_np(msg):
    h, w, step = msg.height, msg.width, msg.step
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, np.uint8).reshape(h, step)
    if enc in ("rgb8", "bgr8"):
        img = buf[:, :w*3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == "rgb8" else img
    if enc == "mono8":
        return buf[:, :w].reshape(h, w)
    raise ValueError(enc)


def depth_to_u16_mm(msg):
    h, w, step = msg.height, msg.width, msg.step
    enc = (msg.encoding or "").lower()
    if enc == "16uc1":
        buf = np.frombuffer(msg.data, np.uint16).reshape(h, step//2)
        return buf[:, :w].reshape(h, w)
    if enc == "32fc1":
        buf = np.frombuffer(msg.data, np.float32).reshape(h, step//4)
        d = buf[:, :w].reshape(h, w)
        out = np.zeros_like(d, np.uint16)
        ok = np.isfinite(d) & (d > 0)
        out[ok] = np.clip(d[ok]*1000, 0, 65535).astype(np.uint16)
        return out
    raise ValueError(enc)


class RGBDOdomSource(Node):
    """
    只负责：
    - 订阅 RGB / Depth / Odom
    - 缓存最近数据
    - 对外提供 get_latest()
    """

    def __init__(
        self,
        rgb_topic,
        depth_topic,
        odom_topic,
        max_dt_ms=50.0,
        cache_size=50,
    ):
        super().__init__("rgbd_odom_source")
        self.max_dt_ns = int(max_dt_ms * 1e6)

        self.depth_cache = deque(maxlen=cache_size)  # (t, depth)
        self.odom_cache  = deque(maxlen=cache_size)  # (t, (x,y,z))
        self.latest_rgb  = None                     # (t, rgb)

        self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 50)
        self.create_subscription(Image, rgb_topic, self._on_rgb, 10)

        self.last_odom = None

    def _on_depth(self, msg):
        self.depth_cache.append(
            (stamp_to_ns(msg.header.stamp), depth_to_u16_mm(msg))
        )

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = (p.x, p.y, p.z)
        quat = (q.x, q.y, q.z, q.w)   # 注意顺序：xyzw
        self.odom_cache.append(
            (stamp_to_ns(msg.header.stamp), (pos, quat))
        )

    def _on_rgb(self, msg):
        self.latest_rgb = (
            stamp_to_ns(msg.header.stamp),
            rgb_to_np(msg)
        )

    @staticmethod
    def _nearest(cache, t):
        if not cache:
            return None
        return min(cache, key=lambda x: abs(x[0] - t))

    def get_latest(self):
        """
        返回最新的同步 RGB/Depth/Odom 数据。
        
        关键特性：
        - 只返回"新的 RGB"数据（RGB时间戳必须更新）
        - Depth/Odom 跟随最新的 RGB 查找（时间同步，允许偏差 max_dt_ns）
        - 重复调用时，如果 RGB 还是老数据，返回 None
        
        返回：
            (t_rgb, rgb, depth, pos, quat) 或 None
        """
        if self.latest_rgb is None:
            return None

        t_rgb, rgb = self.latest_rgb
        
        # === 关键：检查 RGB 时间戳是否真的更新了 ===
        # 只有当 RGB 时间戳变化时，才允许返回新数据
        if hasattr(self, "last_rgb_stamp") and t_rgb == self.last_rgb_stamp:
            return None
        
        dep = self._nearest(self.depth_cache, t_rgb)
        odom = self._nearest(self.odom_cache, t_rgb)

        if dep is None or odom is None:
            return None
        if abs(dep[0] - t_rgb) > self.max_dt_ns:
            return None
        if abs(odom[0] - t_rgb) > self.max_dt_ns:
            return None

        pos, quat = odom[1]

        self.last_rgb_stamp = t_rgb
        return t_rgb, rgb, dep[1], pos, quat
