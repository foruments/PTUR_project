#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Доработанная версия: добавлена визуализация keypoints, bounding box и debug-изображения.
"""
import numpy as np
import time
import cv2
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

KP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7,    "right_elbow": 8,
    "left_wrist": 9,    "right_wrist": 10,
}

class HandTrackerNode(Node):
    def __init__(self):
        super().__init__("hand_tracker_node")
        
        # Параметры (без изменений)
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "link_0")
        self.declare_parameter("model_path", "yolov8n-pose.pt")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("kp_conf_threshold", 0.5)
        self.declare_parameter("depth_scale", 0.001)      
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

        self.get_logger().info("="*40)
        self.get_logger().info(f"📷 Топики:")
        self.get_logger().info(f"   Color: {g('color_topic').value}")
        self.get_logger().info(f"   Depth: {g('depth_topic').value}")
        self.get_logger().info(f"🤖 Base Frame: {self.base_frame}")
        self.get_logger().info("="*40)

        # YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal("Не найден пакет ultralytics. pip3 install ultralytics")
            raise
        self.model = YOLO(g("model_path").value)
        self.device = g("device").value
        self.get_logger().info(f"🧠 YOLOv8-pose: {g('model_path').value} ({self.device})")

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.intr = None
        self._object_present = False
        
        # Счетчики
        self.color_count = 0
        self.depth_count = 0
        self.sync_count = 0
        self.last_log_time = time.time()

        # Издатели (добавлены новые)
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/debug_markers", 10)
        self.debug_image_pub = self.create_publisher(Image, "~/debug_image", 10)  # НОВОЕ

        # Подписки
        self.create_subscription(CameraInfo, g("camera_info_topic").value,
                                 self.on_camera_info, qos_profile_sensor_data)
        self.create_subscription(Image, g("color_topic").value, self.cb_color, qos_profile_sensor_data)
        self.create_subscription(Image, g("depth_topic").value, self.cb_depth, qos_profile_sensor_data)

        color_sub = message_filters.Subscriber(self, Image, g("color_topic").value,
                                               qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, g("depth_topic").value,
                                               qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.on_frames)
        
        self.get_logger().info("✅ Нода запущена, ожидаю кадры…")

    def cb_color(self, msg):
        self.color_count += 1

    def cb_depth(self, msg):
        self.depth_count += 1

    def on_camera_info(self, msg: CameraInfo):
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        if fx <= 0.0 or fy <= 0.0:
            return
        if self.intr is None:
            self.get_logger().info(f"📷 Интринсики: fx={fx:.1f}, fy={fy:.1f}")
        self.intr = (fx, fy, cx, cy)

    def on_frames(self, color_msg: Image, depth_msg: Image):
        self.sync_count += 1
        current_time = time.time()
        
        if current_time - self.last_log_time >= 2.0:
            self.get_logger().info(f"📥 За 2 сек: Color={self.color_count}, Depth={self.depth_count}, Sync={self.sync_count}")
            self.color_count = 0
            self.depth_count = 0
            self.last_log_time = current_time

        if self.intr is None:
            return

        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge: {e}")
            return

        depth_unit = 1.0 if depth_msg.encoding == "32FC1" else self.depth_scale
        stamp = color_msg.header.stamp
        
        # YOLO inference
        start_infer = time.time()
        results = self.model.predict(color, device=self.device, verbose=False)
        infer_time = time.time() - start_infer
        
        if self.sync_count % 30 == 0:
            self.get_logger().info(f"⏱️ YOLO: {infer_time*1000:.0f} мс")

        # Создаем копию изображения для отладки
        debug_img = color.copy()
        
        if not results or not results[0].keypoints or not results[0].keypoints.xy:
            if self.sync_count % 30 == 0:
                self.get_logger().info("👤 Люди не найдены")
            self.clear_object(stamp)
            self.publish_debug_image(debug_img, [])
            return

        kpts = results[0].keypoints
        xy = kpts.xy.cpu().numpy()
        conf = (kpts.conf.cpu().numpy() if kpts.conf is not None
                else np.ones(xy.shape[:2]))

        candidates = []
        all_detections = []  # Для визуализации
        
        for person_idx, person_xy in enumerate(xy):
            person_conf = conf[person_idx]
            
            for side in ("left", "right"):
                arm = self.process_arm(person_xy, person_conf, side,
                                       depth, depth_unit, stamp)
                if arm is not None:
                    candidates.append(arm)
                    # Сохраняем для визуализации
                    i_sh = KP[f"{side}_shoulder"]
                    i_el = KP[f"{side}_elbow"]
                    i_wr = KP[f"{side}_wrist"]
                    all_detections.append({
                        'shoulder': person_xy[i_sh],
                        'elbow': person_xy[i_el],
                        'wrist': person_xy[i_wr],
                        'side': side,
                        'conf': person_conf[[i_sh, i_el, i_wr]]
                    })

        if not candidates:
            if self.sync_count % 30 == 0:
                self.get_logger().info("✋ Руки не найдены (низкий confidence)")
            self.clear_object(stamp)
            self.publish_debug_image(debug_img, all_detections)
            return

        # Выбираем ближайшую руку к рабочей зоне
        best, _ = select_nearest(candidates, self.workzone_center)
        self.publish_capsule(best[0], best[1], stamp)
        
        # Публикуем расширенную визуализацию
        self.publish_detailed_markers(all_detections, stamp)
        self.publish_debug_image(debug_img, all_detections)

    def process_arm(self, person_xy, person_conf, side, depth, depth_unit, stamp):
        i_sh, i_el, i_wr = KP[f"{side}_shoulder"], KP[f"{side}_elbow"], KP[f"{side}_wrist"]
        
        # Проверка: локоть должен быть виден с высокой уверенностью
        if person_conf[i_el] < self.kp_conf_th:
            return None
            
        elbow_cam = self.pixel_to_camera(person_xy[i_el], depth, depth_unit)
        if elbow_cam is None:
            return None
            
        wrist_visible = person_conf[i_wr] >= self.kp_conf_th
        wrist_cam = (self.pixel_to_camera(person_xy[i_wr], depth, depth_unit)
                     if wrist_visible else None)
                     
        if wrist_cam is None:
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
                tf2_ros.ConnectivityException):
            return None
        out = do_transform_point(ps, tf)
        return np.array([out.point.x, out.point.y, out.point.z], dtype=float)

    def publish_capsule(self, elbow, wrist, stamp):
        # Без изменений - публикация в MoveIt
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

    def publish_detailed_markers(self, detections, stamp):
        """Публикация keypoints и bounding box для отладки"""
        arr = MarkerArray()
        
        for idx, det in enumerate(detections):
            # Ключевые точки (сферы)
            points = [
                ('shoulder', det['shoulder'], (1.0, 0.0, 0.0)),  # Красный
                ('elbow', det['elbow'], (0.0, 1.0, 0.0)),        # Зеленый
                ('wrist', det['wrist'], (0.0, 0.0, 1.0))         # Синий
            ]
            
            for kp_idx, (name, uv, color) in enumerate(points):
                marker = Marker()
                marker.header.frame_id = self.camera_optical_frame
                marker.header.stamp = stamp
                marker.ns = f"keypoints_{idx}"
                marker.id = kp_idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.scale.x = marker.scale.y = marker.scale.z = 0.03
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = (*color, 0.8)
                marker.pose.orientation.w = 1.0
                
                # Депроекция точки
                if self.intr:
                    fx, fy, cx, cy = self.intr
                    u, v = int(uv[0]), int(uv[1])
                    # Используем примерную глубину (будет неточно, но для визуализации достаточно)
                    z = 0.5  # Примерно 50см
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    marker.pose.position.x = float(x)
                    marker.pose.position.y = float(y)
                    marker.pose.position.z = float(z)
                    arr.markers.append(marker)
            
            # Bounding box вокруг предплечья (линии)
            bbox_marker = Marker()
            bbox_marker.header.frame_id = self.camera_optical_frame
            bbox_marker.header.stamp = stamp
            bbox_marker.ns = f"bbox_{idx}"
            bbox_marker.id = 100 + idx
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD
            bbox_marker.scale.x = 0.005  # Толщина линии
            bbox_marker.color.r = 1.0
            bbox_marker.color.g = 0.5
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = 0.8
            
            # Добавляем точки локтя и запястья как вершины bbox
            for kp_name in ['elbow', 'wrist']:
                uv = det[kp_name]
                if self.intr:
                    fx, fy, cx, cy = self.intr
                    u, v = int(uv[0]), int(uv[1])
                    z = 0.5
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    p = Pose()
                    p.position.x = float(x)
                    p.position.y = float(y)
                    p.position.z = float(z)
                    p.orientation.w = 1.0
                    bbox_marker.points.append(p.position)
            
            arr.markers.append(bbox_marker)
        
        self.marker_pub.publish(arr)

    def publish_debug_image(self, color, detections):
        """Публикация изображения с наложенными bbox и скелетами"""
        for det in detections:
            # Рисуем ключевые точки
            for kp_name, color_rgb in [('shoulder', (0, 0, 255)), 
                                        ('elbow', (0, 255, 0)), 
                                        ('wrist', (255, 0, 0))]:
                uv = det[kp_name]
                cv2.circle(color, (int(uv[0]), int(uv[1])), 5, color_rgb, -1)
            
            # Рисуем линии скелета
            cv2.line(color, 
                     (int(det['shoulder'][0]), int(det['shoulder'][1])),
                     (int(det['elbow'][0]), int(det['elbow'][1])),
                     (0, 255, 255), 2)
            cv2.line(color,
                     (int(det['elbow'][0]), int(det['elbow'][1])),
                     (int(det['wrist'][0]), int(det['wrist'][1])),
                     (0, 255, 255), 2)
            
            # Bounding box вокруг предплечья
            points = [det['elbow'], det['wrist']]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, y1 = int(min(x_coords) - 30), int(min(y_coords) - 30)
            x2, y2 = int(max(x_coords) + 30), int(max(y_coords) + 30)
            cv2.rectangle(color, (x1, y1), (x2, y2), (255, 128, 0), 2)
            
            # Подпись
            cv2.putText(color, f"{det['side']} arm", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
        
        msg = self.bridge.cv2_to_imgmsg(color, encoding="bgr8")
        self.debug_image_pub.publish(msg)

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
