#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Чистая геометрия алгоритма 5.1 (без зависимостей от ROS).
Вынесено отдельно, чтобы покрыть юнит-тестами и проверять без робота.
"""
import math
import numpy as np


def quaternion_from_z_to(direction):
    """Кватернион (x, y, z, w), поворачивающий локальную ось +Z в вектор direction."""
    v = np.asarray(direction, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    v = v / n
    z = np.array([0.0, 0.0, 1.0])
    d = float(np.dot(z, v))
    if d > 0.999999:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(z, v)
    s = math.sqrt((1.0 + d) * 2.0)
    return (axis[0] / s, axis[1] / s, axis[2] / s, s / 2.0)


def deproject_pixel(u, v, z, fx, fy, cx, cy):
    """(u, v) пиксель + глубина z (м) + интринсики -> точка (X, Y, Z) в СК камеры."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)


def extrapolate_wrist(elbow, shoulder, forearm_length):
    """Оценка запястья при окклюзии: от локтя вдоль вектора (локоть - плечо)."""
    elbow = np.asarray(elbow, dtype=float)
    shoulder = np.asarray(shoulder, dtype=float)
    direction = elbow - shoulder
    n = np.linalg.norm(direction)
    if n < 1e-6:
        return None
    return elbow + (direction / n) * float(forearm_length)


def capsule_geometry(elbow, wrist):
    """Возвращает (центр_цилиндра, высота, кватернион) для капсулы локоть->запястье."""
    elbow = np.asarray(elbow, dtype=float)
    wrist = np.asarray(wrist, dtype=float)
    axis = wrist - elbow
    height = float(np.linalg.norm(axis))
    mid = 0.5 * (elbow + wrist)
    quat = quaternion_from_z_to(axis)
    return mid, height, quat


def select_nearest(candidates, center):
    """Из списка рук [(elbow, wrist), ...] выбирает ближайшую к center (по центру капсулы)."""
    center = np.asarray(center, dtype=float)
    best, best_dist = None, None
    for elbow, wrist in candidates:
        mid = 0.5 * (np.asarray(elbow, float) + np.asarray(wrist, float))
        dist = float(np.linalg.norm(mid - center))
        if best is None or dist < best_dist:
            best, best_dist = (elbow, wrist), dist
    return best, best_dist
