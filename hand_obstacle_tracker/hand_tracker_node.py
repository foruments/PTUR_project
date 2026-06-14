#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Алгоритм 5.1 — обработка данных камеры.

Нода: принимает цветной кадр + выровненную глубину с Intel RealSense
(realsense-ros, ROS 2 Jazzy), запускает YOLOv8-pose, выбирает "рабочую" руку,
обрабатывает окклюзию кисти, переводит пиксели+глубину в 3D и по TF2 в базу
робота, строит капсулу (цилиндр + 2 сферы) и публикует её в MoveIt2 как
CollisionObject (PlanningScene, is_diff=True), а также как Marker для RViz.

Чистая математика — в модуле geometry.py (покрыта юнит-тестами).
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from .geometry import (deproject_pixel, extrapolate_wrist,
                       capsule_geometry, select_nearest)

# Индексы keypoints YOLOv8-pose (COCO-17)
KP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,    "right_elbow": 8,
    "left_wrist": 9,    "right_wrist": 10,
}


class HandTrackerNode(Node):
    def __init__(self):
        super().__init__("hand_tracker_node")

        # ---------- Параметры ----------
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "link_0")
        self.declare_parameter("model_path", "yolov8n-pose.pt")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("kp_conf_threshold", 0.5)
        self.declare_parameter("depth_scale", 0.001)      # uint16 мм -> метры
        self.declare_parameter("depth_window", 5)
        self.declare_parameter("arm_radius", 0.06)
        self.declare_parameter("safety_margin", 0.05)
        self.declare_parameter("forearm_length", 0.25)
        self.declare_parameter("object_id", "human_arm")
        self.declare_parameter("workzone_center", [0.0, 0.0, 0.0])

        g = self.get_parameter
        self.camera_optical_frame = g("camera_optical_frame").value
        self.base_frame = g("base_frame").value
        self.kp_conf_th = float(g("kp_conf_threshold").value)
        self.depth_scale = float(g("depth_scale").value)
        self.depth_window = int(g("depth_window").value)
        self.arm_radius = float(g("arm_radius").value)
        self.safety_margin = float(g("safety_margin").value)
        self.forearm_length = float(g("forearm_length").value)
        self.object_id = g("object_id").value
        self.workzone_center = np.array(g("workzone_center").value, dtype=float)

        # ---------- YOLO ----------
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal(
                "Не найден пакет ultralytics. Установите: pip3 install ultralytics")
            raise
        self.model = YOLO(g("model_path").value)
        self.device = g("device").value
        self.get_logger().info(f"YOLOv8-pose загружена: {g('model_path').value} ({self.device})")

        # ---------- Вспомогательное ----------
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.intr = None             # (fx, fy, cx, cy) — кэш интринсик
        self._object_present = False

        # ---------- Издатели ----------
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/debug_markers", 10)

        # ---------- Подписки ----------
        # camera_info кэшируем отдельно (sensor_data QoS совместим и с reliable,
        # и с best_effort издателем — частая причина "нет данных" при mismatch).
        self.create_subscription(CameraInfo, g("camera_info_topic").value,
                                 self.on_camera_info, qos_profile_sensor_data)
        color_sub = message_filters.Subscriber(self, Image, g("color_topic").value,
                                                qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, g("depth_topic").value,
                                               qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_frames)

        self.get_logger().info("hand_tracker_node запущена, ожидаю кадры…")

    # ----------------------------------------------------------------- #
    def on_camera_info(self, msg: CameraInfo):
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn("camera_info: некорректные интринсики (fx/fy = 0)",
                                   throttle_duration_sec=5.0)
            return
        self.intr = (fx, fy, cx, cy)

    # ----------------------------------------------------------------- #
    def on_frames(self, color_msg: Image, depth_msg: Image):
        if self.intr is None:
            self.get_logger().warn("Жду camera_info…", throttle_duration_sec=5.0)
            return
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}", throttle_duration_sec=2.0)
            return

        # Глубина: 16UC1 -> мм (умножаем на depth_scale); 32FC1 -> уже метры
        depth_unit = 1.0 if depth_msg.encoding == "32FC1" else self.depth_scale
        stamp = color_msg.header.stamp

        results = self.model.predict(color, device=self.device, verbose=False)
        if not results:
            self.clear_object(stamp)
            return
        kpts = results[0].keypoints
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            self.clear_object(stamp)
            return

        xy = kpts.xy.cpu().numpy()
        conf = (kpts.conf.cpu().numpy() if kpts.conf is not None
                else np.ones(xy.shape[:2]))

        candidates = []
        for person in range(xy.shape[0]):
            for side in ("left", "right"):
                arm = self.process_arm(xy[person], conf[person], side,
                                       depth, depth_unit, stamp)
                if arm is not None:
                    candidates.append(arm)

        if not candidates:
            self.clear_object(stamp)
            return

        best, _ = select_nearest(candidates, self.workzone_center)
        self.publish_capsule(best[0], best[1], stamp)

    # ----------------------------------------------------------------- #
    def process_arm(self, person_xy, person_conf, side, depth, depth_unit, stamp):
        i_sh, i_el, i_wr = KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]
        if person_conf[i_el] < self.kp_conf_th:
            return None

        elbow_cam = self.pixel_to_camera(person_xy[i_el], depth, depth_unit)
        if elbow_cam is None:
            return None

        wrist_visible = person_conf[i_wr] >= self.kp_conf_th
        wrist_cam = (self.pixel_to_camera(person_xy[i_wr], depth, depth_unit)
                     if wrist_visible else None)

        if wrist_cam is None:  # окклюзия кисти -> экстраполяция от локтя
            if person_conf[i_sh] < self.kp_conf_th:
                return None
            shoulder_cam = self.pixel_to_camera(person_xy[i_sh], depth, depth_unit)
            if shoulder_cam is None:
                return None
            wrist_cam = extrapolate_wrist(elbow_cam, shoulder_cam, self.forearm_length)
            if wrist_cam is None:
                return None

        elbow_b = self.to_base(elbow_cam, stamp)
        wrist_b = self.to_base(wrist_cam, stamp)
        if elbow_b is None or wrist_b is None:
            return None
        return elbow_b, wrist_b

    # ----------------------------------------------------------------- #
    def pixel_to_camera(self, uv, depth, depth_unit):
        fx, fy, cx, cy = self.intr
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        z = self.sample_depth(depth, u, v, depth_unit)
        if z is None or z <= 0.0:
            return None
        return deproject_pixel(u, v, z, fx, fy, cx, cy)

    def sample_depth(self, depth, u, v, depth_unit):
        r = self.depth_window // 2
        h, w = depth.shape[:2]
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth[v0:v1, u0:u1].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid)) * depth_unit

    def to_base(self, point_cam, stamp):
        ps = PointStamped()
        ps.header.frame_id = self.camera_optical_frame
        ps.header.stamp = stamp
        ps.point.x, ps.point.y, ps.point.z = map(float, point_cam)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_optical_frame, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(
                f"TF {self.base_frame}<-{self.camera_optical_frame}: {e}",
                throttle_duration_sec=2.0)
            return None
        out = do_transform_point(ps, tf)
        return np.array([out.point.x, out.point.y, out.point.z], dtype=float)

    # ----------------------------------------------------------------- #
    def publish_capsule(self, elbow, wrist, stamp):
        radius = self.arm_radius + self.safety_margin
        mid, height, (qx, qy, qz, qw) = capsule_geometry(elbow, wrist)

        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.header.stamp = stamp
        obj.id = self.object_id
        obj.operation = CollisionObject.ADD

        if height > 1e-3:
            cyl = SolidPrimitive()
            cyl.type = SolidPrimitive.CYLINDER
            cyl.dimensions = [height, radius]
            cp = Pose()
            cp.position.x, cp.position.y, cp.position.z = map(float, mid)
            cp.orientation.x, cp.orientation.y, cp.orientation.z, cp.orientation.w = qx, qy, qz, qw
            obj.primitives.append(cyl)
            obj.primitive_poses.append(cp)

        for end in (elbow, wrist):
            sph = SolidPrimitive()
            sph.type = SolidPrimitive.SPHERE
            sph.dimensions = [radius]
            sp = Pose()
            sp.position.x, sp.position.y, sp.position.z = map(float, end)
            sp.orientation.w = 1.0
            obj.primitives.append(sph)
            obj.primitive_poses.append(sp)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        self.scene_pub.publish(scene)
        self._object_present = True
        self.publish_marker(mid, (qx, qy, qz, qw), height, radius, elbow, wrist, stamp)

    def clear_object(self, stamp):
        if not self._object_present:
            return
        obj = CollisionObject()
        obj.header.frame_id = self.base_frame
        obj.header.stamp = stamp
        obj.id = self.object_id
        obj.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)
        self.scene_pub.publish(scene)
        self._object_present = False

    # ----------------------------------------------------------------- #
    def publish_marker(self, mid, quat, height, radius, elbow, wrist, stamp):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns = "human_arm"; m.id = 0
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, mid)
        (m.pose.orientation.x, m.pose.orientation.y,
         m.pose.orientation.z, m.pose.orientation.w) = quat
        m.scale.x = m.scale.y = 2.0 * radius
        m.scale.z = max(height, 1e-3)
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 0.5
        arr.markers.append(m)
        for idx, end in enumerate((elbow, wrist), start=1):
            s = Marker()
            s.header.frame_id = self.base_frame
            s.header.stamp = stamp
            s.ns = "human_arm"; s.id = idx
            s.type = Marker.SPHERE; s.action = Marker.ADD
            s.pose.position.x, s.pose.position.y, s.pose.position.z = map(float, end)
            s.pose.orientation.w = 1.0
            s.scale.x = s.scale.y = s.scale.z = 2.0 * radius
            s.color.r, s.color.g, s.color.b, s.color.a = 1.0, 0.3, 0.0, 0.5
            arr.markers.append(s)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
