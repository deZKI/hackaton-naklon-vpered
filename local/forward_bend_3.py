#!/usr/bin/env python3
"""
Forward‑Bend Monitor v2
=======================
Контролёр упражнения «Наклон вперёд на тумбе»
Доработки включают:
1. Конфиг‑файл (YAML) или параметры CLI.
2. Аудио‑фидбэк для событий (старт удержания, успех, сброс).
3. Прогресс‑бар удержания на экране.
4. Логирование действий в файл.
5. Сохранение GIF (или MP4) короткого клипа вместо пары PNG.
6. Автокалибровка порога прямого пальца по первым кадрам.
7. Обработка исключений и graceful shutdown.
8. Выбор backend RTM‑детектора и fallback на CPU.

Зависимости (pip install):
    opencv‑python, numpy, pyyaml, simpleaudio, imageio, rtmlib

Пример `config.yaml` рядом со скриптом:
------------------------------------------------
backend: onnxruntime        # onnxruntime | torch
device: cuda                # cuda | cpu
kpt_threshold: 0.5
straight_knee_threshold: 140
hold_duration: 2.0          # seconds
finger_px_threshold: 60     # initial, авто‑тюнинг включён
results_dir: results
save_format: mp4            # gif | mp4 | png
log_file: monitor.log
output_duration: 3.0          # seconds length
------------------------------------------------
CLI‑параметры перекрывают YAML → значения по умолчанию.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

try:
    import simpleaudio as sa  # cross‑platform WAV playback
except ImportError:
    sa = None  # fallback later

try:
    from rtmlib import Wholebody
    # Будем использовать локальную версию draw_skeleton_safe (ниже)
except ImportError as e:
    print("❌ Не найден rtmlib:", e)
    sys.exit(1)

# ---------- Константы ключевых точек ----------
RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST = 6, 8, 10
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 12, 14, 16
LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST = 5, 7, 9
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 11, 13, 15
LEFT_INDEX_MCP, LEFT_INDEX_TIP = 112, 115
RIGHT_INDEX_MCP, RIGHT_INDEX_TIP = 91, 94

# ---------- Звуки ----------
SOUND_START, SOUND_SUCCESS, SOUND_RESET = "start.wav", "success.wav", "reset.wav"


def play_sound(path: str):
    if not sa:
        return  # простой fallback (нет simpleaudio)
    p = Path(__file__).with_name(path)
    if p.exists():
        try:
            wave_obj = sa.WaveObject.from_wave_file(str(p))
            wave_obj.play()
        except Exception as exc:
            logging.warning("Audio playback failed: %s", exc)


# ---------- Утилиты расчёта углов / дистанций ----------

def calculate_angle(a, b, c):
    a, b, c = map(np.asarray, (a, b, c))
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def angle_by_kpts(scores, kps, i1, i2, i3, thr):
    if (scores[i1] > thr) and (scores[i2] > thr) and (scores[i3] > thr):
        return calculate_angle(kps[i1], kps[i2], kps[i3])
    return None


def knees_angles(scores, kps, thr):
    left = angle_by_kpts(scores, kps, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, thr)
    right = angle_by_kpts(scores, kps, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, thr)
    return left, right


# ---------- Локальная функция отрисовки скелета, безопасная для любого количества точек ----------

def draw_skeleton_safe(frame, keypoints, scores, thr=0.5):
    """Простая отрисовка: точки + линии для первых 17 связей COCO.
    Работает даже если keypoints > цветов.
    """
    base_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    # draw points
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > thr:
            color = base_colors[i % len(base_colors)]
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    # basic skeleton (COCO upper-body) – отрисуем, только если индексы в диапазоне
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]
    for i1, i2 in connections:
        if i1 < len(keypoints) and i2 < len(keypoints):
            if scores[i1] > thr and scores[i2] > thr:
                pt1 = tuple(map(int, keypoints[i1]))
                pt2 = tuple(map(int, keypoints[i2]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)


# ---------- Основной класс контролёра ----------

class ForwardBendMonitor:
    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.state = "WAIT_START"  # WAIT_START → HOLD → DONE
        self.hold_start: float | None = None
        self.fps = 30  # target FPS, используется для видеобуфера
        self.output_frames = int(cfg.output_duration * self.fps)
        self.buffer: deque[np.ndarray] = deque(maxlen=int(cfg.output_duration * self.fps * 1.5))
        self.auto_thresh_samples: list[float] = []
        self.auto_calibrated = False
        self.finger_px_threshold = cfg.finger_px_threshold
        self.output_frames = int(cfg.output_duration * self.fps)
        self.wrist_ref_y = None  # для контроля неподвижности руки
        self.wrist_tolerance = 20  # px допуск движения
        self._init_video()
        self._init_logger()
        self.body = Wholebody(mode="lightweight", backend=cfg.backend, device=cfg.device)
        logging.info("Detector initialised: backend=%s, device=%s", cfg.backend, cfg.device)

    # ----- initialisation helpers -----
    def _init_video(self):
        # Cam 0 – side, Cam 1 – front
        self.cap_side = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap_front = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        for cap in (self.cap_side, self.cap_front):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap_side.isOpened() or not self.cap_front.isOpened():
            logging.error("Couldn't open cameras")
            raise RuntimeError("Camera init failed")

    def _init_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.cfg.log_file, encoding="utf‑8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info("===== Session started =====")

    # ----- Core helpers -----
    def _save_clip(self, frames: list[np.ndarray], prefix: str):
        # Берём только последние output_duration секунд
        if len(frames) > self.output_frames:
            frames = frames[-self.output_frames:]
        ts = datetime.now().strftime("%Y‑%m‑%d_%H‑%M‑%S")
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
        else:  # fallback png pair
            for idx, fr in enumerate((frames[0], frames[-1])):
                fname = out_dir / f"{prefix}_{ts}_{idx}.png"
                cv2.imwrite(str(fname), fr)
        logging.info("Saved result to %s", fp)

    def _draw_progress(self, frame, progress):
        # draw filled rectangle (bottom, width 300px)
        h, w = frame.shape[:2]
        bar_w, bar_h = 300, 20
        x0, y0 = (w - bar_w) // 2, h - bar_h - 10
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 2)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), (0, 255, 0), -1)
        cv2.putText(frame, f"{int(progress * 100)}%", (x0 + bar_w + 10, y0 + bar_h - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----- Hand / finger check with auto‑tune -----
    def _fingers_ok(self, scores, kps, frame):
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
            # авто‑калибровка
            if not self.auto_calibrated and len(self.auto_thresh_samples) < 30:
                self.auto_thresh_samples.append(dist)
                if len(self.auto_thresh_samples) == 30:
                    median = float(np.median(self.auto_thresh_samples))
                    self.finger_px_threshold = max(40.0, 0.8 * median)
                    self.auto_calibrated = True
                    logging.info("Finger threshold auto‑set to %.1f px", self.finger_px_threshold)
            finger_straight = dist > self.finger_px_threshold
            color = (0, 255, 0) if finger_straight else (0, 0, 255)
            cv2.line(frame, tuple(map(int, kps[mcp_idx])), tuple(map(int, kps[tip_idx])), color, 2)
            cv2.putText(frame, f"{label}:{int(dist)}", tuple(map(int, kps[tip_idx])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if finger_straight:
                ok = True
        return ok

    # ----- Main loop -----
    def run(self):
        logging.info("▶ Monitoring started: hold %.1f s", self.cfg.hold_duration)
        play_sound(SOUND_START)
        try:
            while True:
                t0 = time.time()
                ret_s, frm_s = self.cap_side.read()
                ret_f, frm_f = self.cap_front.read()
                if not ret_s or not ret_f:
                    logging.error("Frame capture failed")
                    break

                # Side
                kps_s, scr_s = self.body(frm_s)
                if np.ndim(kps_s) == 3:
                    kps_s, scr_s = kps_s[0], scr_s[0]

                # Нарисуем скелет на боковом кадре
                draw_skeleton_safe(frm_s, kps_s, scr_s, self.cfg.kpt_threshold)

                l_ang, r_ang = knees_angles(scr_s, kps_s, self.cfg.kpt_threshold)
                knees_ok = False
                if (l_ang is not None) and (r_ang is not None):
                    knees_ok = (l_ang >= self.cfg.straight_knee_threshold) or (
                        r_ang >= self.cfg.straight_knee_threshold)
                # annotate angles
                if l_ang is not None:
                    cv2.putText(frm_s, f"LK:{l_ang:.0f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0) if l_ang >= self.cfg.straight_knee_threshold else (0, 0, 255), 2)
                if r_ang is not None:
                    cv2.putText(frm_s, f"RK:{r_ang:.0f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0) if r_ang >= self.cfg.straight_knee_threshold else (0, 0, 255), 2)

                # Проверка, что видна хотя бы одна стопа
                foot_ok = False
                if (LEFT_ANKLE < len(scr_s) and scr_s[LEFT_ANKLE] > self.cfg.kpt_threshold):
                    foot_ok = True
                elif (RIGHT_ANKLE < len(scr_s) and scr_s[RIGHT_ANKLE] > self.cfg.kpt_threshold):
                    foot_ok = True

                # Front
                kps_f, scr_f = self.body(frm_f)
                if np.ndim(kps_f) == 3:
                    kps_f, scr_f = kps_f[0], scr_f[0]
                draw_skeleton_safe(frm_f, kps_f, scr_f, self.cfg.kpt_threshold)
                hands_ok = self._fingers_ok(scr_f, kps_f, frm_f)

                # Координаты запястий для контроля неподвижности
                curr_wrist_y = None
                if LEFT_WRIST < len(scr_f) and scr_f[LEFT_WRIST] > self.cfg.kpt_threshold:
                    curr_wrist_y = kps_f[LEFT_WRIST][1]
                elif RIGHT_WRIST < len(scr_f) and scr_f[RIGHT_WRIST] > self.cfg.kpt_threshold:
                    curr_wrist_y = kps_f[RIGHT_WRIST][1]

                # FSM
                if self.state == "WAIT_START":
                    if knees_ok and hands_ok and foot_ok:
                        self.state = "HOLD"
                        self.hold_start = time.time()
                        self.wrist_ref_y = curr_wrist_y
                        play_sound(SOUND_START)
                        logging.info("Hold started")
                elif self.state == "HOLD":
                    wrist_move_ok = True
                    if self.wrist_ref_y is not None and curr_wrist_y is not None:
                        if abs(curr_wrist_y - self.wrist_ref_y) > self.wrist_tolerance:
                            wrist_move_ok = False

                    if not knees_ok or not hands_ok or not foot_ok or not wrist_move_ok:
                        self.state = "WAIT_START"
                        self.hold_start = None
                        self.wrist_ref_y = None
                        self.buffer.clear()
                        play_sound(SOUND_RESET)
                        logging.info("Hold reset")
                    else:
                        # progress
                        elapsed = time.time() - self.hold_start
                        progress = min(1.0, elapsed / self.cfg.hold_duration)
                        self._draw_progress(frm_f, progress)
                        if progress >= 1.0:
                            play_sound(SOUND_SUCCESS)
                            logging.info("Success! Saving clip…")
                            self._save_clip(list(self.buffer), "bend")
                            # готовимся к новой попытке
                            self.state = "WAIT_START"
                            self.buffer.clear()
                            self.hold_start = None
                            logging.info("Ready for next attempt")
                elif self.state == "DONE":
                    pass  # wait for user to quit or press R to restart

                # draw state label
                cv2.putText(frm_s, f"State:{self.state}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 0), 2)

                if knees_ok:
                    cv2.putText(frm_s, "Knees:STRAIGHT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frm_s, "Knees:BENT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if foot_ok:
                    cv2.putText(frm_s, "Foot:VISIBLE", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    cv2.putText(frm_s, "Foot:NOT", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                # show windows
                cv2.imshow("Side", frm_s)
                cv2.imshow("Front", frm_f)

                # ----------- Side-by-side кадр для GIF -----------
                # Убедимся, что обе половины одинаковой высоты
                if frm_s.shape[0] != frm_f.shape[0]:
                    # подгоняем высоту фронт-кадра под боковой
                    new_w = int(frm_f.shape[1] * frm_s.shape[0] / frm_f.shape[0])
                    frm_f_resized = cv2.resize(frm_f, (new_w, frm_s.shape[0]))
                else:
                    frm_f_resized = frm_f
                combined = np.hstack((frm_s, frm_f_resized))

                # push combined frame to buffer
                self.buffer.append(combined.copy())

                # ----------- Таймкод -----------
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                for frame in (frm_s, frm_f):
                    cv2.putText(frame, ts, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                key = cv2.waitKey(max(1, int(1000 / self.fps - (time.time() - t0) * 1000))) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r") and self.state == "DONE":
                    logging.info("Restarting after success …")
                    self.state = "WAIT_START"
                    self.buffer.clear()

        finally:
            self.cap_side.release()
            self.cap_front.release()
            cv2.destroyAllWindows()
            logging.info("Session ended")


# ---------- Конфигурация & запуск ----------

def load_cfg(path: str) -> SimpleNamespace:
    default = dict(
        backend="onnxruntime",
        device="cuda",
        kpt_threshold=0.5,
        straight_knee_threshold=140,
        hold_duration=2.0,
        finger_px_threshold=60.0,
        results_dir="results",
        save_format="mp4",
        log_file="monitor.log",
        output_duration=5.0,
    )
    if Path(path).exists():
        with open(path, "r", encoding="utf‑8") as f:
            user_cfg = yaml.safe_load(f) or {}
        default.update(user_cfg)
    return SimpleNamespace(**default)


def parse_args():
    p = argparse.ArgumentParser(description="Forward‑Bend posture monitor")
    p.add_argument("--config", default="config.yaml", help="YAML config path")
    p.add_argument("--backend", choices=["onnxruntime", "torch"], help="Override backend")
    p.add_argument("--device", choices=["cpu", "cuda"], help="Override device")
    p.add_argument("--output-duration", type=float, help="Seconds length of saved GIF/MP4")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    if args.backend:
        cfg.backend = args.backend
    if args.device:
        cfg.device = args.device
    if args.output_duration:
        cfg.output_duration = args.output_duration

    # пересчитать output_frames в соответствии с новым cfg
    monitor = ForwardBendMonitor(cfg)
    monitor.run()


if __name__ == "__main__":
    main()
