import cv2
import numpy as np
from rtmlib import Body
from enum import Enum

# --- Константы и нормативы (из stages.js) ---
stages = [
    {
        'step': 7,
        'start_age': 18,
        'end_age': 19,
        'quantity_man': {'bronze': 25, 'silver': 32, 'gold': 43},
        'quantity_woman': {'bronze': 8, 'silver': 12, 'gold': 17}
    },
    {
        'step': 8,
        'start_age': 20,
        'end_age': 24,
        'quantity_man': {'bronze': 27, 'silver': 33, 'gold': 45},
        'quantity_woman': {'bronze': 9, 'silver': 13, 'gold': 18}
    },
    {
        'step': 9,
        'start_age': 25,
        'end_age': 29,
        'quantity_man': {'bronze': 21, 'silver': 25, 'gold': 40},
        'quantity_woman': {'bronze': 8, 'silver': 12, 'gold': 17}
    },
    {
        'step': 10,
        'start_age': 30,
        'end_age': 34,
        'quantity_man': {'bronze': 15, 'silver': 19, 'gold': 33},
        'quantity_woman': {'bronze': 5, 'silver': 8, 'gold': 14}
    },
]

# --- Пользователь ---
class User:
    def __init__(self, name='KIRILL', age = 17, sex='мужской', uid=17):
        self.name = name
        self.age = age
        self.sex = sex
        self.uid = uid
        self.stage = self.find_stage()

    def find_stage(self):
        for stage in stages:
            if stage['start_age'] <= self.age <= stage['end_age']:
                return stage
        return None

# --- Параметры позы и подсчёт отжиманий ---
class PositionEnum(Enum):
    UP = 1
    DOWN = 2
    UNKNOWN = 3

KPT_THRESHOLD = 0.5
MIN_LIMIT_VALUE_HIP = 160
MAX_LIMIT_VALUE_HIP = 180
MAX_LIMIT_VALUE_ELBOW_DOWN = 85
MAX_LIMIT_VALUE_ELBOW_UP = 165

# Индексы COCO whole-body
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

skeleton_connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

keypoint_colors = [
    (0,0,255)]*5 + [(255,0,0)]*6 + [(0,255,0)]*6


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_angle_by_keypoints(scores, kp, i1, i2, i3):
    if (scores[i1] > KPT_THRESHOLD and scores[i2] > KPT_THRESHOLD and scores[i3] > KPT_THRESHOLD):
        return calculate_angle(kp[i1], kp[i2], kp[i3])
    return None

def detected_position(angle_elbow, angle_hip):
    if angle_elbow and angle_hip:
        hip_normalize = MIN_LIMIT_VALUE_HIP <= angle_hip <= MAX_LIMIT_VALUE_HIP
        elbow_bent = angle_elbow <= MAX_LIMIT_VALUE_ELBOW_DOWN
        elbow_straight = angle_elbow >= MAX_LIMIT_VALUE_ELBOW_UP
        if elbow_bent and hip_normalize:
            return PositionEnum.DOWN
        elif elbow_straight and hip_normalize:
            return PositionEnum.UP
    return PositionEnum.UNKNOWN

def draw_skeleton(frame, keypoints, scores):
    for i, (x, y) in enumerate(keypoints):
        if scores[i] > KPT_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 4, keypoint_colors[i], -1)
    for i1, i2 in skeleton_connections:
        if scores[i1] > KPT_THRESHOLD and scores[i2] > KPT_THRESHOLD:
            pt1 = tuple(map(int, keypoints[i1]))
            pt2 = tuple(map(int, keypoints[i2]))
            cv2.line(frame, pt1, pt2, (0,255,255), 2)

def get_result(pushups, user):
    if not user.stage:
        return 'Ступень не найдена'
    norms = user.stage['quantity_man'] if user.sex == 'Мужской' else user.stage['quantity_woman']
    if pushups >= norms['gold']:
        return 'Золото 🥇'
    elif pushups >= norms['silver']:
        return 'Серебро 🥈'
    elif pushups >= norms['bronze']:
        return 'Бронза 🥉'
    else:
        return 'Не сдал ❌'

def main():

    user = User()
    print(f'Ступень: {user.stage}')

    # --- Модель ---
    wholebody = Body(mode='lightweight', backend='onnxruntime', device='cpu')

    # --- Камеры ---
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(2)
    POS_UP = PositionEnum.UP
    POS_DOWN = PositionEnum.DOWN
    POS_UNKNOWN = PositionEnum.UNKNOWN
    state = POS_UNKNOWN
    pushup_count = 0
    positions = []

    print('Нажмите Q для завершения.')
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print('Ошибка чтения кадров!')
            break
        # Можно выбрать одну из камер для анализа, либо обе (пример: только frame1)
        frame = frame1  # или frame2, или объединить
        keypoints, scores = wholebody(frame)
        if isinstance(keypoints, (list, np.ndarray)) and np.array(keypoints).ndim == 3:
            kp = keypoints[0]
            sc = scores[0]
        else:
            kp = keypoints
            sc = scores
        # Подсчёт углов
        left_elbow_angle = calculate_angle_by_keypoints(sc, kp, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
        left_hip_angle = calculate_angle_by_keypoints(sc, kp, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
        position = detected_position(left_elbow_angle, left_hip_angle)
        positions.append(position)
        # Логика подсчёта отжиманий (как в main.js)
        if position == POS_UNKNOWN:
            pass
        elif state == POS_UNKNOWN and position == POS_UP:
            state = POS_UP
        elif state == POS_UP and position == POS_DOWN:
            state = POS_DOWN
        elif state == POS_DOWN and position == POS_UP:
            pushup_count += 1
            print(f'Отжиманий: {pushup_count}')
            state = POS_UP
        # Визуализация
        draw_skeleton(frame, kp, sc)
        cv2.putText(frame, f'Push-ups: {pushup_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Camera 1', frame1)
        cv2.imshow('Camera 2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    print(f'Итого отжиманий: {pushup_count}')
    print('Результат:', get_result(pushup_count, user))

if __name__ == '__main__':
    main()
