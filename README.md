# hand_obstacle_tracker — алгоритм 5.1 (трекинг руки → капсула-препятствие)

Часть проекта «Динамический учёт препятствий в рабочей зоне iiwa».
Нода берёт поток с Intel RealSense, находит руку человека (YOLOv8-pose), строит
вокруг предплечья капсулу с защитным зазором и публикует её в MoveIt2 как
`CollisionObject`. MoveIt2 в режиме `allow_replanning` обходит эту капсулу.

Целевое окружение: **ROS 2 Jazzy**, `realsense-ros`, MoveIt2, `lbr_fri_ros2_stack`.

---

## 1. Зависимости

```bash
# ROS-пакеты
sudo apt install ros-jazzy-realsense2-camera ros-jazzy-cv-bridge \
                 ros-jazzy-tf2-ros ros-jazzy-tf2-geometry-msgs \
                 ros-jazzy-message-filters ros-jazzy-moveit-msgs \
                 ros-jazzy-vision-msgs

# Python (в том же окружении, где запускается ROS)
pip3 install ultralytics opencv-python numpy
```

Модель `yolov8n-pose.pt` скачается автоматически при первом запуске
(или положите её рядом и укажите путь в `config/params.yaml`).

---

## 2. Сборка

```bash
cd ~/ros2_ws/src
# скопируйте сюда папку hand_obstacle_tracker
cd ~/ros2_ws
colcon build --packages-select hand_obstacle_tracker
source install/setup.bash
```

---

## 3. Запуск камеры (отдельным терминалом)

```bash
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=true \
    align_depth.enable:=true \
    rgb_camera.profile:=640x480x30 \
    depth_module.profile:=640x480x30
```

Проверьте, что идут топики:
```bash
ros2 topic list | grep camera
# .../color/image_raw
# .../aligned_depth_to_color/image_raw
# .../color/camera_info
```

---

## 4. Калибровка eye-to-hand (один раз, до запуска ноды)

Камера на штативе → калибровка «глаз вне руки». На фланец крепится маркер
(нужно напечатать держатель). Утилита: `easy_handeye2` или `moveit2_calibration`.

Результат калибровки публикуется как статический TF
`base_frame → camera_link`. Без корректного TF нода не сможет перевести
координаты руки в систему робота. Проверка дерева кадров:

```bash
ros2 run tf2_tools view_frames
# затем впишите реальные имена кадров в config/params.yaml:
#   camera_optical_frame  (обычно camera_color_optical_frame)
#   base_frame            (link_0 или lbr_link_0 — зависит от URDF)
```

---

## 5. Запуск ноды трекинга

```bash
# только нода (камера уже запущена отдельно)
ros2 launch hand_obstacle_tracker hand_tracking.launch.py

# с GPU
ros2 launch hand_obstacle_tracker hand_tracking.launch.py device:=cuda:0

# поднять камеру тем же файлом
ros2 launch hand_obstacle_tracker hand_tracking.launch.py launch_camera:=true
```

Визуализация в RViz: добавьте дисплей `MarkerArray` на топик
`/hand_tracker_node/debug_markers` (оранжевая капсула повторяет руку) и
`PlanningScene`, чтобы видеть объект `human_arm` в MoveIt.

---

## 6. Как это работает (конвейер 5.1)

1. Синхронный приём: `color/image_raw` + `aligned_depth_to_color/image_raw` + `color/camera_info`.
2. YOLOv8-pose → 17 keypoints на человека.
3. Для каждой руки берём плечо (5/6), локоть (7/8), запястье (9/10).
4. Окклюзия: если запястье не видно (низкая уверенность / нет глубины) —
   экстраполируем от локтя вдоль вектора `локоть → плечо` на `forearm_length`.
5. Депроекция: `(u,v)` + глубина → `(X,Y,Z)` в оптической СК камеры
   (`X=(u-cx)·Z/fx`, `Y=(v-cy)·Z/fy`).
6. TF2: `camera_optical_frame → base_frame`.
7. Капсула = цилиндр `локоть→запястье` + 2 сферы, радиус `arm_radius + safety_margin`.
8. Публикация `PlanningScene(is_diff=True)` с `CollisionObject` в `/planning_scene`.
   Если рука пропала — объект удаляется (`REMOVE`).

Если людей в кадре несколько — выбирается рука, ближайшая к `workzone_center`
(точка отсчёта в системе координат робота).

---

## 7. Что ещё предстоит (вне этого пакета)

- Связать выход (capsule/CollisionObject) с перепланировщиком (задача 5.2).
- Геометрическая маскировка самого робота в облаке точек (FCL + URDF) —
  чтобы звенья робота не принимались за препятствие.
- Перенос критичных по времени узлов на C++ при необходимости.
- Тонкая настройка `arm_radius`, `safety_margin`, порогов под реальную сцену.
