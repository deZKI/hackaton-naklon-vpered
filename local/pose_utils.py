from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# ---------- Константы ключевых точек (COCO + MediaPipe Whole Body) ----------
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
LEFT_INDEX_MCP, LEFT_INDEX_TIP = 112, 115
RIGHT_INDEX_MCP, RIGHT_INDEX_TIP = 91, 94

__all__ = [
    # keypoints
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "RIGHT_ANKLE",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
    "LEFT_HIP",
    "LEFT_KNEE",
    "LEFT_ANKLE",
    "LEFT_INDEX_MCP",
    "LEFT_INDEX_TIP",
    "RIGHT_INDEX_MCP",
    "RIGHT_INDEX_TIP",
    # funcs
    "calculate_angle",
    "angle_by_kpts",
    "knees_angles",
    "draw_skeleton_safe",
]


# ---------- Геометрия ----------

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Возвращает угол ABC в градусах."""
    a, b, c = map(np.asarray, (a, b, c))
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(angle)


def angle_by_kpts(
    scores: np.ndarray,
    kps: np.ndarray,
    i1: int,
    i2: int,
    i3: int,
    thr: float,
) -> float | None:
    """Вычисляет угол по трём индексам ключевых точек при достаточной уверенности."""
    if (scores[i1] > thr) and (scores[i2] > thr) and (scores[i3] > thr):
        return calculate_angle(kps[i1], kps[i2], kps[i3])
    return None


def knees_angles(scores: np.ndarray, kps: np.ndarray, thr: float) -> Tuple[float | None, float | None]:
    """Возвращает углы левого и правого колена."""
    left = angle_by_kpts(scores, kps, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, thr)
    right = angle_by_kpts(scores, kps, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, thr)
    return left, right


# ---------- Визуализация ----------

def draw_skeleton_safe(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> None:
    """Отрисовка ключевых точек и базовых связей (COCO) с учётом переменного числа точек."""
    base_colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    # точки
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > thr:
            color = base_colors[i % len(base_colors)]
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # линии (верхняя часть COCO)
    connections = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    for i1, i2 in connections:
        if i1 < len(keypoints) and i2 < len(keypoints):
            if scores[i1] > thr and scores[i2] > thr:
                pt1 = tuple(map(int, keypoints[i1]))
                pt2 = tuple(map(int, keypoints[i2]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1) 