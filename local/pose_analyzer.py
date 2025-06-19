from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

import logging

import cv2
import numpy as np

from rtmlib import Wholebody

from config import MonitorConfig
from pose_utils import (
    LEFT_ANKLE,
    LEFT_INDEX_MCP,
    LEFT_INDEX_TIP,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_INDEX_MCP,
    RIGHT_INDEX_TIP,
    RIGHT_WRIST,
    draw_skeleton_safe,
    knees_angles,
)

@dataclass
class SideAnalysis:
    kps: np.ndarray
    scores: np.ndarray
    knees_ok: bool
    l_ang: Optional[float]
    r_ang: Optional[float]
    foot_ok: bool


@dataclass
class FrontAnalysis:
    kps: np.ndarray
    scores: np.ndarray
    hands_ok: bool
    curr_wrist_y: Optional[float]


__all__ = ["PoseAnalyzer", "SideAnalysis", "FrontAnalysis"]


class PoseAnalyzer:
    """Инкапсулирует анализ позы: детекция ключевых точек, проверка коленей/пальцев.

    Возвращает готовые структуры с результатами для дальнейшей обработки FSM.
    """

    def __init__(self, detector: Wholebody, cfg: MonitorConfig) -> None:
        self.detector = detector
        self.cfg = cfg

        # данные автокалибровки порога «прямого» пальца
        self.auto_thresh_samples: list[float] = []
        self.auto_calibrated: bool = False
        self.finger_px_threshold: float = cfg.finger_px_threshold

    # ------------------------------------------------------------------
    def _fingers_ok(self, scores: np.ndarray, kps: np.ndarray, frame: np.ndarray) -> bool:
        ok = False
        for mcp_idx, tip_idx, label in [
            (LEFT_INDEX_MCP, LEFT_INDEX_TIP, "L"),
            (RIGHT_INDEX_MCP, RIGHT_INDEX_TIP, "R"),
        ]:
            if tip_idx >= len(scores):
                continue
            if scores[mcp_idx] < self.cfg.kpt_threshold or scores[tip_idx] < self.cfg.kpt_threshold:
                continue

            dist = np.linalg.norm(np.asarray(kps[mcp_idx]) - np.asarray(kps[tip_idx]))
            # авто-калибровка
            if not self.auto_calibrated and len(self.auto_thresh_samples) < 30:
                self.auto_thresh_samples.append(dist)
                if len(self.auto_thresh_samples) == 30:
                    median = float(np.median(self.auto_thresh_samples))
                    self.finger_px_threshold = max(40.0, 0.8 * median)
                    self.auto_calibrated = True
                    logging.info("Finger threshold auto-set to %.1f px", self.finger_px_threshold)

            finger_straight = dist > self.finger_px_threshold
            color = (0, 255, 0) if finger_straight else (0, 0, 255)
            cv2.line(frame, tuple(map(int, kps[mcp_idx])), tuple(map(int, kps[tip_idx])), color, 2)
            cv2.putText(
                frame,
                f"{label}:{int(dist)}",
                tuple(map(int, kps[tip_idx])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            if finger_straight:
                ok = True
        return ok

    # ------------------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Запускает детектор и приводит вывод к двум массивам 2-D."""
        kps, scr = self.detector(frame)
        if np.ndim(kps) == 3:
            kps, scr = kps[0], scr[0]
        return kps, scr

    # ------------------------------------------------------------------
    def analyze_side(self, frame: np.ndarray) -> SideAnalysis:
        """Анализ кадра боковой камеры."""
        kps, scr = self._detect(frame)
        draw_skeleton_safe(frame, kps, scr, self.cfg.kpt_threshold)

        # колени
        l_ang, r_ang = knees_angles(scr, kps, self.cfg.kpt_threshold)
        knees_ok = True
        if l_ang is not None and l_ang < self.cfg.straight_knee_threshold:
            knees_ok = False
        if r_ang is not None and r_ang < self.cfg.straight_knee_threshold:
            knees_ok = False

        # стопа: хотя бы одна видна
        foot_ok = False
        if LEFT_ANKLE < len(scr) and scr[LEFT_ANKLE] > self.cfg.kpt_threshold:
            foot_ok = True
        elif RIGHT_ANKLE < len(scr) and scr[RIGHT_ANKLE] > self.cfg.kpt_threshold:
            foot_ok = True

        return SideAnalysis(kps, scr, knees_ok, l_ang, r_ang, foot_ok)

    # ------------------------------------------------------------------
    def analyze_front(self, frame: np.ndarray) -> FrontAnalysis:
        """Анализ кадра фронтальной камеры."""
        kps, scr = self._detect(frame)
        draw_skeleton_safe(frame, kps, scr, self.cfg.kpt_threshold)

        hands_ok = self._fingers_ok(scr, kps, frame)

        # координата запястья (любого, которое видно)
        curr_wrist_y: float | None = None
        if LEFT_WRIST < len(scr) and scr[LEFT_WRIST] > self.cfg.kpt_threshold:
            curr_wrist_y = kps[LEFT_WRIST][1]
        elif RIGHT_WRIST < len(scr) and scr[RIGHT_WRIST] > self.cfg.kpt_threshold:
            curr_wrist_y = kps[RIGHT_WRIST][1]

        return FrontAnalysis(kps, scr, hands_ok, curr_wrist_y) 