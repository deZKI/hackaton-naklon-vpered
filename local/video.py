from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

__all__ = ["DualCamera"]


class DualCamera:
    """Простая оболочка над двумя `cv2.VideoCapture`.

    Используется в `ForwardBendMonitor` для чтения кадров с боковой и фронтальной камер.
    """

    def __init__(
        self,
        side_idx: int = 0,
        front_idx: int = 1,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.cap_side = cv2.VideoCapture(side_idx, cv2.CAP_DSHOW)
        self.cap_front = cv2.VideoCapture(front_idx, cv2.CAP_DSHOW)

        for cap in (self.cap_side, self.cap_front):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap_side.isOpened() or not self.cap_front.isOpened():
            logging.error("Couldn't open cameras")
            raise RuntimeError("Camera init failed")

        fps_prop = self.cap_side.get(cv2.CAP_PROP_FPS)
        self.fps: int = int(fps_prop) if fps_prop and fps_prop > 0 else 30
        if fps_prop == 0 or fps_prop is None:
            logging.warning("Camera FPS unavailable, defaulting to 30 FPS")

    # ---------------------------------------------------------------------
    def grab(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Читает кадры, возвращает флаг успеха и два кадра (side, front)."""
        ret_s, frm_s = self.cap_side.read()
        ret_f, frm_f = self.cap_front.read()
        return ret_s and ret_f, frm_s, frm_f

    # ---------------------------------------------------------------------
    def release(self) -> None:
        """Освобождает обе камеры."""
        self.cap_side.release()
        self.cap_front.release() 