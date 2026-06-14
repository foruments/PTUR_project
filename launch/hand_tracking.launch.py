#!/usr/bin/env python3
"""
Запуск алгоритма 5.1.

По умолчанию поднимает только ноду трекинга (камеру удобнее запускать отдельно,
чтобы не перезапускать драйвер). Чтобы поднять камеру тем же launch-файлом,
передайте launch_camera:=true (требуется установленный пакет realsense2_camera).

Пример:
  ros2 launch hand_obstacle_tracker hand_tracking.launch.py
  ros2 launch hand_obstacle_tracker hand_tracking.launch.py launch_camera:=true device:=cuda:0
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory("hand_obstacle_tracker")
    default_params = os.path.join(pkg_share, "config", "params.yaml")

    params_file = LaunchConfiguration("params_file")
    launch_camera = LaunchConfiguration("launch_camera")
    device = LaunchConfiguration("device")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="YAML с параметрами ноды трекинга"),
        DeclareLaunchArgument("launch_camera", default_value="false",
                              description="Поднять realsense2_camera этим же файлом"),
        DeclareLaunchArgument("device", default_value="cpu",
                              description="Устройство инференса YOLO: cpu | cuda:0"),
    ]

    camera = GroupAction(
        condition=IfCondition(launch_camera),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(PathJoinSubstitution(
                    [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"])),
                launch_arguments={
                    "pointcloud.enable": "true",
                    "align_depth.enable": "true",
                    "rgb_camera.profile": "640x480x30",
                    "depth_module.profile": "640x480x30",
                }.items(),
            )
        ],
    )

    tracker = Node(
        package="hand_obstacle_tracker",
        executable="hand_tracker_node",
        name="hand_tracker_node",
        output="screen",
        parameters=[params_file, {"device": device}],
    )

    return LaunchDescription(args + [camera, tracker])
