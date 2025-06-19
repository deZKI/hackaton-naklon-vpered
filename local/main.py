import cv2
import numpy as np
from rtmlib import Body
from utils.lib import calculate_angle_by_keypoints, detected_position, PositionEnum, User, LEFT_SHOULDER, \
    LEFT_ELBOW, LEFT_WRIST, get_result, draw_skeleton, LEFT_HIP, LEFT_KNEE  # твои функции


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
