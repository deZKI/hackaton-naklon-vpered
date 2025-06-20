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
import math
from audio_utils import SOUND_RESET, SOUND_START, SOUND_SUCCESS, play_sound
from config import MonitorConfig
from video import DualCamera
from pose_analyzer import PoseAnalyzer

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
        # fps определяется после инициализации DualCamera
        self.fps: int | None = None
        self.output_frames: int | None = None  # заполним позже
        self.buffer: deque[np.ndarray] | None = None  # заполним позже

        # контроль движения запястья
        self.wrist_ref_y: float | None = None
        self.wrist_tolerance = cfg.wrist_tolerance

        # счётчик, используемый в состоянии POST_CAPTURE
        self._post_frames_left: int = 0

        self._init_logger()

        self.body = Wholebody(mode=cfg.mode, backend=cfg.backend, device=cfg.device)
        logging.info("Detector initialised: backend=%s, device=%s", cfg.backend, cfg.device)

        # --- Video ---
        self.camera = DualCamera()
        self.fps = self.camera.fps

        # теперь когда fps известен — создаём буфер вывода
        self.output_frames = int(cfg.output_duration * self.fps)
        self.buffer = deque(maxlen=int(cfg.output_duration * self.fps * 1.5))

        # --- Pose Analyzer ---
        self.analyzer = PoseAnalyzer(self.body, self.cfg)

    # ----- init helpers -----
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
            # Сохраняем финальный кадр PNG
            png_fp = out_dir / f"{prefix}_{ts}_final.png"
            cv2.imwrite(str(png_fp), frames[-1])
        elif self.cfg.save_format == "mp4":
            fp = out_dir / f"{prefix}_{ts}.mp4"
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(fp), fourcc, self.fps, (w, h))
            for fr in frames:
                vw.write(fr)
            vw.release()
            # Сохраняем финальный кадр PNG
            png_fp = out_dir / f"{prefix}_{ts}_final.png"
            cv2.imwrite(str(png_fp), frames[-1])
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

    # ----- video helpers -----
    def _grab_frames(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        """Считывает кадры из обеих камер. Возвращает (успех, side, front)."""
        return self.camera.grab()

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

                # ---------- Pose analysis ----------
                side = self.analyzer.analyze_side(frm_s)
                front = self.analyzer.analyze_front(frm_f)

                knees_ok = side.knees_ok
                l_ang = side.l_ang
                r_ang = side.r_ang
                foot_ok = side.foot_ok

                hands_ok = front.hands_ok
                curr_wrist_y = front.curr_wrist_y

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
                        self.buffer.clear()
                        logging.info("Hold reset")
                    else:
                        elapsed = time.time() - self.hold_start
                        progress = min(1.0, elapsed / self.cfg.hold_duration)
                        self._draw_progress(frm_f, progress)
                        if progress >= 1.0:
                            self._save_clip(list(self.buffer), "bend")
                            # Успех: переходим в POST_CAPTURE, чтобы захватить ещё N кадров
                            play_sound(SOUND_SUCCESS)
                            self._post_frames_left = int(self.fps * self.cfg.post_capture_seconds)
                            self.state = MonitorState.POST_CAPTURE
                elif self.state == MonitorState.POST_CAPTURE:
                    # продолжаем писать кадры до тех пор, пока не соберём требуемое количество
                    self._post_frames_left -= 1
                    if self._post_frames_left <= 0:
                        logging.info("Saving clip after post-capture …")
                        # self._save_clip(list(self.buffer), "bend")
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

                # Получаем размеры изображения
                height, width = frm_f.shape[:2]

                # Задаём координаты двух горизонтальных линий
                y1 = int(height * 0.3)  # линия на 30% от верха
                y2 = int(height)  # линия на 70% от верха

                # Рисуем линии
                cv2.line(frm_f, (0, y1), (width, y1), color=(0, 255, 0), thickness=2)
                cv2.line(frm_f, (0, y2), (width, y2), color=(0, 0, 255), thickness=2)

                height_ruler_sm = 25
                height_ruler_px = height * 0.7

                coef_size = height_ruler_sm / height_ruler_px
                top_line_y = height * 0.3
                max_hand_y = front.high_finger[1]

                result = (max_hand_y - top_line_y) * coef_size
                result = 0 if result < 0 else result

                cv2.putText(frm_f, str(math.ceil(result * 10) / 10) + "sm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            Color.GREEN.value, 2)
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
            self.camera.release()
            if not self.cfg.headless:
                cv2.destroyAllWindows()
            logging.info("Session ended") 