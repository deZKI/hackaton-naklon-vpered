import cv2
import time
import os
import numpy as np
from datetime import datetime

from rtmlib import Wholebody, draw_skeleton

RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10
RIGHT_HIP = 12
RIGHT_KNEE = 14
RIGHT_ANKLE = 16
LEFT_SHOULDER = 5
LEFT_ELBOW = 7
LEFT_WRIST = 9
LEFT_HIP = 11
LEFT_KNEE = 13
LEFT_ANKLE = 15

def calculate_angle_by_keypoints(scores, kp, i1, i2, i3):
    if (scores[i1] > KPT_THRESHOLD and scores[i2] > KPT_THRESHOLD and scores[i3] > KPT_THRESHOLD):
        return calculate_angle(kp[i1], kp[i2], kp[i3])
    return None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)
# Индексы кистей/пальцев (уточните при необходимости под вашу модель)
LEFT_INDEX_MCP = 112
LEFT_INDEX_TIP = 115
RIGHT_INDEX_MCP = 91
RIGHT_INDEX_TIP = 94

KPT_THRESHOLD = 0.5  # порог уверенности для видимости ключевых точек

# --- Параметры упражнения ---
STRAIGHT_KNEE_THRESHOLD = 140  # минимальный угол, при котором колено считаем прямым
HOLD_DURATION = 2.0  # сек, сколько нужно удерживать наклон

# Папка, куда будут сохраняться скриншоты
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def knees_angles(scores, kps):
    """Возвращает кортеж (left_angle, right_angle). Может быть None, если точки не найдены."""
    left = calculate_angle_by_keypoints(scores, kps, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right = calculate_angle_by_keypoints(scores, kps, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    return left, right


def save_screenshot(frame, prefix: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    filename = f"{prefix}_{ts}.png"
    filepath = os.path.join(RESULTS_DIR, filename)
    # Отобразим таймкод на изображении
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(filepath, frame)
    print(f"Скриншот сохранён: {filepath}")


def hands_visible_and_fingers_straight(scores, kps, frame=None):
    """Проверяет, что минимум одна рука видна спереди и палец вытянут (ладонь->кончик линия почти прямая)."""
    visible = False
    for mcp_idx, tip_idx, hand_label in [
        (LEFT_INDEX_MCP, LEFT_INDEX_TIP, "L"),
        (RIGHT_INDEX_MCP, RIGHT_INDEX_TIP, "R"),
    ]:
        if len(scores) <= tip_idx:  # защита, если модель вернула меньше ключевых точек
            continue
        if scores[mcp_idx] > KPT_THRESHOLD and scores[tip_idx] > KPT_THRESHOLD:
            visible = True
            mcp = kps[mcp_idx]
            tip = kps[tip_idx]
            # Евклидово расстояние между ладонью и кончиком пальца
            dist = np.linalg.norm(np.array(mcp) - np.array(tip))
            # Порог – 60 пикселей (примерно). При необходимости подстройте.
            finger_straight = dist > 60
            if frame is not None:
                color = (0, 255, 0) if finger_straight else (0, 0, 255)
                cv2.line(frame, tuple(map(int, mcp)), tuple(map(int, tip)), color, 2)
                cv2.putText(frame, f"{hand_label}:{int(dist)}", tuple(map(int, tip)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if finger_straight:
                return True  # одна рука уже удовлетворяет
    return False


def main():
    print("▶ Запуск контроля упражнения: Наклон вперёд на тумбе")
    body = Wholebody(mode="lightweight", backend="onnxruntime", device="cuda")

    cap_side = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # камера сбоку
    cap_front = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # камера спереди

    # попросим нормальное разрешение
    for cap in (cap_side, cap_front):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap_side.isOpened() or not cap_front.isOpened():
        print("❌ Не удалось открыть одну из камер")
        return

    state = "WAIT_START"  # WAIT_START → HOLD → DONE
    hold_start = None

    while True:
        ret_side, frame_side = cap_side.read()
        ret_front, frame_front = cap_front.read()
        if not ret_side or not ret_front:
            print("❌ Ошибка чтения кадра с камеры")
            break

        # ------------------ Обработка боковой камеры ------------------
        kps_side, scr_side = body(frame_side)
        if isinstance(kps_side, (list, np.ndarray)) and np.array(kps_side).ndim == 3:
            kps_side = kps_side[0]
            scr_side = scr_side[0]

        left_knee_angle, right_knee_angle = knees_angles(scr_side, kps_side)
        knees_ok = False
        if left_knee_angle is not None and right_knee_angle is not None:
            knees_ok = left_knee_angle >= STRAIGHT_KNEE_THRESHOLD or right_knee_angle >= STRAIGHT_KNEE_THRESHOLD

        # Отобразим углы на кадре
        if left_knee_angle is not None:
            cv2.putText(frame_side, f"LK: {left_knee_angle:.0f}°", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if left_knee_angle >= STRAIGHT_KNEE_THRESHOLD else (0,0,255), 2)
        if right_knee_angle is not None:
            cv2.putText(frame_side, f"RK: {right_knee_angle:.0f}°", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0) if right_knee_angle >= STRAIGHT_KNEE_THRESHOLD else (0,0,255), 2)

        # ------------------ Обработка фронтальной камеры (только скелет для наглядности) ------------------
        kps_front, scr_front = body(frame_front)
        if isinstance(kps_front, (list, np.ndarray)) and np.array(kps_front).ndim == 3:
            kps_front = kps_front[0]
            scr_front = scr_front[0]
        draw_skeleton(frame_front, kps_front, scr_front)

        hands_ok = hands_visible_and_fingers_straight(scr_front, kps_front, frame=frame_front)

        # ------------------ Логика состояния ------------------
        if state == "WAIT_START":
            if knees_ok and hands_ok:
                hold_start = time.time()
                state = "HOLD"
        elif state == "HOLD":
            if not knees_ok or not hands_ok:
                state = "WAIT_START"
                hold_start = None
            else:
                if time.time() - hold_start >= HOLD_DURATION:
                    save_screenshot(frame_side.copy(), "side")
                    save_screenshot(frame_front.copy(), "front")
                    print("✅ Зафиксирована поза: прямые колени + руки перед камерой (2 сек.)")
                    state = "DONE"
        elif state == "DONE":
            # Завершаем по нажатию Q
            pass

        # ------------------ Визуализация ------------------
        cv2.putText(frame_side, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if knees_ok:
            cv2.putText(frame_side, "Knees: STRAIGHT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame_side, "Knees: BENT", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Side Camera", frame_side)
        cv2.imshow("Front Camera", frame_front)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap_side.release()
    cap_front.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 