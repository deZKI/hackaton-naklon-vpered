from __future__ import annotations

import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from enum import Enum, auto

import cv2
import imageio.v2 as imageio
import numpy as np

from rtmlib import Wholebody

from audio_utils import SOUND_RESET, SOUND_START, SOUND_SUCCESS, play_sound
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

from config import MonitorConfig

__all__ = ["ForwardBendMonitor"]


# -------------------- FSM State --------------------


class Color(Enum):
    """BGR-цвета, используемые в приложении."""

    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    YELLOW = (255, 255, 0)


class MonitorState(Enum):
    """Возможные состояния конечного автомата."""

    WAIT_START = auto()  # ожидаем корректной исходной позы
    HOLD = auto()  # удерживаем позу в течение заданного времени
    POST_CAPTURE = auto()  # записываем ещё несколько секунд после успеха


class ForwardBendMonitor:
    """Контролёр позы наклона вперёд с двумя камерами."""

    def __init__(self, cfg: MonitorConfig) -> None:
        self.cfg: MonitorConfig = cfg  # сохранение типа для MyPy
        self.state: MonitorState = MonitorState.WAIT_START
        self.hold_start: float | None = None
        # fps будет определён после открытия камеры в `_init_video`
        self.fps: int | None = None  # заполним позже
        self.output_frames: int | None = None  # заполним позже
        self.buffer: deque[np.ndarray] | None = None  # заполним позже

        # авто-калибровка распрямления пальца
        self.auto_thresh_samples: list[float] = []
        self.auto_calibrated = False
        self.finger_px_threshold = cfg.finger_px_threshold

        # контроль движения запястья
        self.wrist_ref_y: float | None = None
        self.wrist_tolerance = cfg.wrist_tolerance

        # счётчик, используемый в состоянии POST_CAPTURE
        self._post_frames_left: int = 0

        self._init_video()  # определяет self.fps

        # теперь когда fps известен — создаём буфер вывода
        self.output_frames = int(cfg.output_duration * self.fps)
        self.buffer = deque(maxlen=int(cfg.output_duration * self.fps * 1.5))

        self._init_logger()

        self.body = Wholebody(mode=cfg.mode, backend=cfg.backend, device=cfg.device)
        logging.info("Detector initialised: backend=%s, device=%s", cfg.backend, cfg.device)

    # ----- init helpers -----
    def _init_video(self) -> None:
        # Cam 0 – side, Cam 1 – front
        self.cap_side = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap_front = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        for cap in (self.cap_side, self.cap_front):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap_side.isOpened() or not self.cap_front.isOpened():
            logging.error("Couldn't open cameras")
            raise RuntimeError("Camera init failed")

        # Попытаемся узнать fps из камеры; если не удалось — fallback к 30.
        fps_prop = self.cap_side.get(cv2.CAP_PROP_FPS)
        self.fps = int(fps_prop) if fps_prop and fps_prop > 0 else 30
        if fps_prop == 0 or fps_prop is None:
            logging.warning("Camera FPS unavailable, defaulting to 30 FPS")

    def _init_logger(self) -> None:
        # гарантируем, что каталог для логов существует
        Path(self.cfg.log_file).resolve().parent.mkdir(exist_ok=True, parents=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.cfg.log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
        logging.info("===== Session started =====")

    # ----- helpers -----
    def _save_clip(self, frames: list[np.ndarray], prefix: str) -> None:
        if len(frames) > self.output_frames:
            frames = frames[-self.output_frames:]
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(self.cfg.results_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        if self.cfg.save_format == "gif":
            fp = out_dir / f"{prefix}_{ts}.gif"
            imageio.mimsave(fp, frames, fps=self.fps)
        elif self.cfg.save_format == "mp4":
            fp = out_dir / f"{prefix}_{ts}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(fp), fourcc, self.fps, (w, h))
            for fr in frames:
                vw.write(fr)
            vw.release()
        else:
            # fallback: сохраним первый и последний кадр PNG
            for idx, fr in enumerate((frames[0], frames[-1])):
                cv2.imwrite(str(out_dir / f"{prefix}_{ts}_{idx}.png"), fr)
        logging.info("Saved result to %s", fp)

    def _draw_progress(self, frame: np.ndarray, progress: float) -> None:
        h, w = frame.shape[:2]
        bar_w, bar_h = 300, 20
        x0, y0 = (w - bar_w) // 2, h - bar_h - 10
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), Color.WHITE.value, 2)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), Color.GREEN.value, -1)
        cv2.putText(
            frame,
            f"{int(progress * 100)}%",
            (x0 + bar_w + 10, y0 + bar_h - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            Color.GREEN.value,
            2,
        )

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
            color = Color.GREEN.value if finger_straight else Color.RED.value
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

    # ----- video helpers -----
    def _grab_frames(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Считывает кадры из обеих камер. Возвращает (успех, side, front)."""
        ret_s, frm_s = self.cap_side.read()
        ret_f, frm_f = self.cap_front.read()
        return ret_s and ret_f, frm_s, frm_f

    # ----- main loop -----
    def run(self) -> None:
        logging.info("▶ Monitoring started: hold %.1f s", self.cfg.hold_duration)
        play_sound(SOUND_START)
        try:
            while True:
                t0 = time.time()
                ok, frm_s, frm_f = self._grab_frames()
                if not ok:
                    logging.error("Frame capture failed")
                    break

                # ---------- Side camera processing ----------
                kps_s, scr_s = self.body(frm_s)
                if np.ndim(kps_s) == 3:
                    kps_s, scr_s = kps_s[0], scr_s[0]

                draw_skeleton_safe(frm_s, kps_s, scr_s, self.cfg.kpt_threshold)

                l_ang, r_ang = knees_angles(scr_s, kps_s, self.cfg.kpt_threshold)
                knees_ok = True
                if l_ang is not None and l_ang < self.cfg.straight_knee_threshold:
                    knees_ok = False
                if r_ang is not None and r_ang < self.cfg.straight_knee_threshold:
                    knees_ok = False

                # annotate angle values
                if l_ang is not None:
                    cv2.putText(
                        frm_s,
                        f"LK:{l_ang:.0f}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        Color.GREEN.value if l_ang >= self.cfg.straight_knee_threshold else Color.RED.value,
                        2,
                    )
                if r_ang is not None:
                    cv2.putText(
                        frm_s,
                        f"RK:{r_ang:.0f}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        Color.GREEN.value if r_ang >= self.cfg.straight_knee_threshold else Color.RED.value,
                        2,
                    )

                # хотя бы одна стопа должна быть видна
                foot_ok = False
                if LEFT_ANKLE < len(scr_s) and scr_s[LEFT_ANKLE] > self.cfg.kpt_threshold:
                    foot_ok = True
                elif RIGHT_ANKLE < len(scr_s) and scr_s[RIGHT_ANKLE] > self.cfg.kpt_threshold:
                    foot_ok = True

                # ---------- Front camera processing ----------
                kps_f, scr_f = self.body(frm_f)
                if np.ndim(kps_f) == 3:
                    kps_f, scr_f = kps_f[0], scr_f[0]
                draw_skeleton_safe(frm_f, kps_f, scr_f, self.cfg.kpt_threshold)
                hands_ok = self._fingers_ok(scr_f, kps_f, frm_f)

                # контроль движения запястья
                curr_wrist_y: float | None = None
                if LEFT_WRIST < len(scr_f) and scr_f[LEFT_WRIST] > self.cfg.kpt_threshold:
                    curr_wrist_y = kps_f[LEFT_WRIST][1]
                elif RIGHT_WRIST < len(scr_f) and scr_f[RIGHT_WRIST] > self.cfg.kpt_threshold:
                    curr_wrist_y = kps_f[RIGHT_WRIST][1]

                # ---------- FSM ----------
                if self.state == MonitorState.WAIT_START:
                    if knees_ok and hands_ok and foot_ok:
                        self.state = MonitorState.HOLD
                        self.hold_start = time.time()
                        self.wrist_ref_y = curr_wrist_y
                        play_sound(SOUND_START)
                        logging.info("Hold started")
                elif self.state == MonitorState.HOLD:
                    wrist_move_ok = True
                    if self.wrist_ref_y is not None and curr_wrist_y is not None:
                        if abs(curr_wrist_y - self.wrist_ref_y) > self.wrist_tolerance:
                            wrist_move_ok = False

                    if not all([knees_ok, hands_ok, foot_ok, wrist_move_ok]):
                        self.state = MonitorState.WAIT_START
                        self.hold_start = None
                        self.wrist_ref_y = None
                        play_sound(SOUND_RESET)
                        logging.info("Hold reset")
                    else:
                        elapsed = time.time() - self.hold_start
                        progress = min(1.0, elapsed / self.cfg.hold_duration)
                        self._draw_progress(frm_f, progress)
                        if progress >= 1.0:
                            # Успех: переходим в POST_CAPTURE, чтобы захватить ещё N кадров
                            play_sound(SOUND_SUCCESS)
                            self._post_frames_left = int(self.fps * self.cfg.post_capture_seconds)
                            self.state = MonitorState.POST_CAPTURE
                elif self.state == MonitorState.POST_CAPTURE:
                    # продолжаем писать кадры до тех пор, пока не соберём требуемое количество
                    self._post_frames_left -= 1
                    if self._post_frames_left <= 0:
                        logging.info("Saving clip after post-capture …")
                        self._save_clip(list(self.buffer), "bend")
                        # подготовка к следующей попытке
                        self.buffer.clear()
                        self.hold_start = None
                        self.state = MonitorState.WAIT_START
                        logging.info("Ready for next attempt")

                # ---------- annotations & display ----------
                cv2.putText(
                    frm_s,
                    f"State:{self.state.name}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    Color.YELLOW.value,
                    2,
                )

                if knees_ok:
                    cv2.putText(frm_s, "Knees:STRAIGHT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.GREEN.value, 2)
                else:
                    cv2.putText(frm_s, "Knees:BENT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.RED.value, 2)

                if foot_ok:
                    cv2.putText(frm_s, "Foot:VISIBLE", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.GREEN.value, 2)
                else:
                    cv2.putText(frm_s, "Foot:NOT", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, Color.RED.value, 2)

                # ----- display (optional) -----
                if not self.cfg.headless:
                    cv2.imshow("Side", frm_s)
                    cv2.imshow("Front", frm_f)

                # side-by-side для сохранения
                if frm_s.shape[0] != frm_f.shape[0]:
                    new_w = int(frm_f.shape[1] * frm_s.shape[0] / frm_f.shape[0])
                    frm_f_resized = cv2.resize(frm_f, (new_w, frm_s.shape[0]))
                else:
                    frm_f_resized = frm_f
                combined = np.hstack((frm_s, frm_f_resized))
                self.buffer.append(combined.copy())

                # таймкод
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                for frame in (frm_s, frm_f):
                    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Color.WHITE.value, 1)

                # ----- FPS / quit handling -----
                delay_ms = max(1, int(1000 / self.fps - (time.time() - t0) * 1000))
                if not self.cfg.headless:
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key == ord("q"):
                        break
                else:
                    # в headless режиме просто ждём указанное время
                    time.sleep(delay_ms / 1000.0)

        finally:
            self.cap_side.release()
            self.cap_front.release()
            if not self.cfg.headless:
                cv2.destroyAllWindows()
            logging.info("Session ended") 