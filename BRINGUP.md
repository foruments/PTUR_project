# BRINGUP — поэтапный запуск и диагностика

Запускать **по этапам** и проверять каждый. Не переходите дальше, пока текущий
этап не прошёл. Так вы локализуете проблему, а не будете искать её во всём сразу.

---

## Этап 0. Сборка и тесты (без железа)
```bash
cd ~/ros2_ws && colcon build --packages-select hand_obstacle_tracker
source install/setup.bash
python3 src/hand_obstacle_tracker/test/test_geometry.py   # должны пройти 10/10
```
✅ Готово, если пакет собрался и тесты прошли.

## Этап 1. Камера публикует данные
```bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true \
    align_depth.enable:=true rgb_camera.profile:=640x480x30 depth_module.profile:=640x480x30
ros2 topic hz /camera/camera/color/image_raw          # должна идти ~30 Гц
ros2 topic echo /camera/camera/color/camera_info --once | grep -A1 "k:"   # fx,fy,cx,cy != 0
```
✅ Идут color, aligned_depth_to_color, camera_info.
⚠️ Если имена топиков другие — впишите их в `config/params.yaml`.

## Этап 2. TF между камерой и роботом (калибровка eye-to-hand)
```bash
ros2 run tf2_tools view_frames           # PDF дерева кадров
ros2 run tf2_ros tf2_echo link_0 camera_color_optical_frame   # должно резолвиться
```
✅ Есть связный путь base_frame → camera_optical_frame.
⚠️ Нет пути → калибровка не опубликована или имена кадров не совпадают с
`params.yaml` (`base_frame`, `camera_optical_frame`). Это самая частая причина,
по которой "ничего не работает".

## Этап 3. Детекция руки (без робота)
```bash
ros2 launch hand_obstacle_tracker hand_tracking.launch.py
ros2 topic echo /hand_tracker_node/debug_markers --once   # появляются при руке в кадре
```
RViz: Fixed Frame = `base_frame`, добавьте `MarkerArray` на
`/hand_tracker_node/debug_markers`. Оранжевая капсула должна повторять руку.
⚠️ Капсула смещена → ошибка калибровки (этап 2) или неверный `depth_scale`.
⚠️ YOLO тормозит на CPU → `device:=cuda:0` или модель `yolov8n-pose`.

## Этап 4. Препятствие видно в MoveIt2
Запущен `move_group` (из вашей конфигурации MoveIt2 для iiwa). В RViz добавьте
дисплей `PlanningScene`. Объект `human_arm` должен появляться в сцене.
```bash
ros2 topic echo /planning_scene --once    # видно world.collision_objects
```

## Этап 5. Обход (вне этого пакета)
Робот обходит капсулу только если в вашем `move_group` включён `allow_replanning`
и настроено исполнение траектории. Это задачи 5.2/5.3 и общая интеграция —
данный пакет лишь поставляет препятствие.

---

## Частые проблемы

| Симптом | Причина | Что делать |
|---------|---------|-----------|
| `Жду camera_info…` в логе | не приходит camera_info | проверьте `camera_info_topic`, этап 1 |
| `TF base<-camera: …` варнинг | нет калибровки/неверные кадры | этап 2, имена в `params.yaml` |
| капсула смещена от руки | плохая калибровка / `depth_scale` | переснять калибровку; для 32FC1 глубина уже в метрах |
| нет детекций | YOLO/свет/расстояние | проверить кадр, снизить `kp_conf_threshold` |
| объект не влияет на план | MoveIt2 не подписан / нет replanning | этапы 4–5, конфигурация move_group |
