from setuptools import find_packages, setup
import os
from glob import glob

package_name = "hand_obstacle_tracker"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Взглядов З. Е.",
    maintainer_email="vzgladovzahar@gmail.com",
    description="Трекинг руки RealSense + YOLOv8-pose -> капсула-препятствие в MoveIt2 (алгоритм 5.1).",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hand_tracker_node = hand_obstacle_tracker.hand_tracker_node:main",
        ],
    },
)
