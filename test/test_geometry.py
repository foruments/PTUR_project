#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Юнит-тесты чистой геометрии алгоритма 5.1 (запускаются без ROS и без камеры).
Запуск:  pytest -q   или   python3 test/test_geometry.py
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hand_obstacle_tracker.geometry import (  # noqa: E402
    quaternion_from_z_to, deproject_pixel, extrapolate_wrist,
    capsule_geometry, select_nearest)


def _rotate(q, p):
    x, y, z, w = q
    u = np.array([x, y, z], float)
    p = np.array(p, float)
    return p + 2.0 * np.cross(u, np.cross(u, p) + w * p)


def test_quaternion_aligns_z_axis():
    for d in ([1, 0, 0], [0, 1, 0], [1, 1, 1], [-1, 2, 0.5], [0, 0, 1], [0, 0, -1]):
        q = quaternion_from_z_to(d)
        r = _rotate(q, [0, 0, 1])
        dn = np.array(d, float) / np.linalg.norm(d)
        assert np.linalg.norm(r - dn) < 1e-6, f"dir={d} -> {r}"


def test_quaternion_zero_vector():
    assert quaternion_from_z_to([0, 0, 0]) == (0.0, 0.0, 0.0, 1.0)


def test_deproject_principal_point():
    # Пиксель в главной точке -> только по оси Z
    p = deproject_pixel(320, 240, 2.0, fx=600, fy=600, cx=320, cy=240)
    assert np.allclose(p, [0.0, 0.0, 2.0])


def test_deproject_known_offset():
    # 1 px вправо от центра при fx=600, Z=3 -> X = 1*3/600 = 0.005
    p = deproject_pixel(321, 240, 3.0, fx=600, fy=600, cx=320, cy=240)
    assert abs(p[0] - 0.005) < 1e-9
    assert abs(p[1] - 0.0) < 1e-9
    assert abs(p[2] - 3.0) < 1e-9


def test_extrapolate_wrist_direction_and_length():
    shoulder = [0, 0, 0]
    elbow = [1, 0, 0]
    wrist = extrapolate_wrist(elbow, shoulder, forearm_length=0.5)
    # продолжение от локтя в ту же сторону, что плечо->локоть, на 0.5 м
    assert np.allclose(wrist, [1.5, 0.0, 0.0])


def test_extrapolate_wrist_degenerate():
    assert extrapolate_wrist([1, 1, 1], [1, 1, 1], 0.3) is None


def test_capsule_geometry_basic():
    mid, height, quat = capsule_geometry([0, 0, 0], [0, 0, 1])
    assert np.allclose(mid, [0, 0, 0.5])
    assert abs(height - 1.0) < 1e-9
    # ось уже вдоль +Z -> единичный кватернион
    assert np.allclose(quat, [0, 0, 0, 1])


def test_capsule_height_diagonal():
    mid, height, _ = capsule_geometry([0, 0, 0], [3, 4, 0])
    assert abs(height - 5.0) < 1e-9
    assert np.allclose(mid, [1.5, 2.0, 0.0])


def test_select_nearest_picks_closest():
    far = (np.array([2, 0, 0.]), np.array([2, 0, 1.]))
    near = (np.array([0.1, 0, 0.]), np.array([0.1, 0, 0.2]))
    best, dist = select_nearest([far, near], center=[0, 0, 0])
    assert np.allclose(best[0], near[0]) and np.allclose(best[1], near[1])
    assert dist < 0.3


def test_select_nearest_empty():
    best, dist = select_nearest([], center=[0, 0, 0])
    assert best is None and dist is None


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} тестов пройдено.")


if __name__ == "__main__":
    _run_all()
