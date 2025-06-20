import cv2
import numpy as np
import time
import pygame
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import rtmlib
from rtmlib import Wholebody

# Инициализация pygame для звука
pygame.mixer.init()

class PullupState(Enum):
    """Состояния выполнения подтягивания"""
    INITIAL = "initial"      # Начальное положение
    PULLING = "pulling"      # Подтягивание
    FINAL = "final"          # Финальное положение
    LOWERING = "lowering"    # Опускание

@dataclass
class PullupConfig:
    """Конфигурация для подсчета подтягиваний"""
    # Пороговые значения углов
    initial_elbow_angle_min: float = 160.0  # Минимальный угол в локтях для начального положения
    final_elbow_angle_max: float = 90.0     # Максимальный угол в локтях для финального положения
    knee_angle_min: float = 160.0           # Минимальный угол в коленях (ноги прямые)
    hip_angle_min: float = 160.0            # Минимальный угол таза (прямое положение)
    
    # Пороговые значения для уверенности в ключевых точках
    confidence_threshold: float = 0.6
    
    # Время удержания финального положения
    hold_duration: float = 0.5
    
    # Пороговые значения для определения подбородка над перекладиной
    chin_bar_distance_threshold: float = 30.0  # пиксели
    
    # Звуковые файлы
    rep_complete_sound: str = "audio/success.mp3"

class PullupCounter:
    """Класс для подсчета повторений подтягиваний с использованием rtmlib"""
    
    def __init__(self, config: PullupConfig):
        self.config = config
        
        # Инициализация rtmlib для отслеживания позы
        self.pose_tracker = Wholebody(mode='lightweight', backend='onnxruntime', device='cpu')
        
        self.state = PullupState.INITIAL
        self.rep_count = 0
        self.hold_start_time = None
        self.last_rep_time = 0
        
        # Загружаем звук
        self.load_sound()
    
    def load_sound(self):
        """Загружает звуковой файл"""
        try:
            pygame.mixer.music.load(self.config.rep_complete_sound)
        except:
            print(f"Failed to load sound: {self.config.rep_complete_sound}")
    
    def play_rep_sound(self):
        """Воспроизводит звук завершения повторения"""
        try:
            pygame.mixer.music.play()
        except:
            pass
    
    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Вычисляет угол между тремя точками"""
        a, b, c = map(np.asarray, (a, b, c))
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return float(angle)
    
    def get_landmark_coords(self, landmarks, landmark_id: int) -> Optional[Tuple[float, float]]:
        """Получает координаты ключевой точки из rtmlib"""
        if landmarks and len(landmarks) > landmark_id:
            point = landmarks[landmark_id]
            if point[2] > self.config.confidence_threshold:  # confidence score
                return (point[0], point[1])  # x, y coordinates
        return None
    
    def analyze_pose(self, landmarks: np.ndarray) -> dict:
        """Анализирует позу и возвращает ключевые параметры"""
        if not landmarks or len(landmarks) < 33:  # rtmlib COCO format has 17 keypoints
            return {}
        
        # rtmlib использует COCO формат ключевых точек
        # Индексы ключевых точек в COCO формате
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_ELBOW = 7
        RIGHT_ELBOW = 8
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        # Получаем координаты ключевых точек
        left_shoulder = self.get_landmark_coords(landmarks, LEFT_SHOULDER)
        left_elbow = self.get_landmark_coords(landmarks, LEFT_ELBOW)
        left_wrist = self.get_landmark_coords(landmarks, LEFT_WRIST)
        left_hip = self.get_landmark_coords(landmarks, LEFT_HIP)
        left_knee = self.get_landmark_coords(landmarks, LEFT_KNEE)
        left_ankle = self.get_landmark_coords(landmarks, LEFT_ANKLE)
        
        right_shoulder = self.get_landmark_coords(landmarks, RIGHT_SHOULDER)
        right_elbow = self.get_landmark_coords(landmarks, RIGHT_ELBOW)
        right_wrist = self.get_landmark_coords(landmarks, RIGHT_WRIST)
        right_hip = self.get_landmark_coords(landmarks, RIGHT_HIP)
        right_knee = self.get_landmark_coords(landmarks, RIGHT_KNEE)
        right_ankle = self.get_landmark_coords(landmarks, RIGHT_ANKLE)
        
        nose = self.get_landmark_coords(landmarks, NOSE)
        
        analysis = {}
        
        # Вычисляем углы в локтях
        if left_shoulder and left_elbow and left_wrist:
            left_elbow_angle = self.calculate_angle(
                np.array(left_shoulder), 
                np.array(left_elbow), 
                np.array(left_wrist)
            )
            analysis['left_elbow_angle'] = left_elbow_angle
        
        if right_shoulder and right_elbow and right_wrist:
            right_elbow_angle = self.calculate_angle(
                np.array(right_shoulder), 
                np.array(right_elbow), 
                np.array(right_wrist)
            )
            analysis['right_elbow_angle'] = right_elbow_angle
        
        # Вычисляем углы в коленях
        if left_hip and left_knee and left_ankle:
            left_knee_angle = self.calculate_angle(
                np.array(left_hip), 
                np.array(left_knee), 
                np.array(left_ankle)
            )
            analysis['left_knee_angle'] = left_knee_angle
        
        if right_hip and right_knee and right_ankle:
            right_knee_angle = self.calculate_angle(
                np.array(right_hip), 
                np.array(right_knee), 
                np.array(right_ankle)
            )
            analysis['right_knee_angle'] = right_knee_angle
        
        # Вычисляем угол таза (между плечами и бедрами)
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Центр плеч
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2, 
                              (left_shoulder[1] + right_shoulder[1]) / 2)
            # Центр бедер
            hip_center = ((left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2)
            
            # Вычисляем угол таза относительно вертикали
            # Используем точку выше плеч для создания вертикальной линии
            vertical_point = (shoulder_center[0], shoulder_center[1] - 50)  # 50 пикселей выше плеч
            
            hip_angle = self.calculate_angle(
                np.array(vertical_point),
                np.array(shoulder_center),
                np.array(hip_center)
            )
            analysis['hip_angle'] = hip_angle
        
        # Определяем положение подбородка относительно плеч (приблизительно)
        if nose and left_shoulder and right_shoulder:
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            chin_position = nose[1]  # Подбородок примерно на уровне носа
            analysis['chin_above_shoulders'] = chin_position < shoulder_y
        
        return analysis
    
    def is_initial_position(self, analysis: dict) -> bool:
        """Проверяет, находится ли человек в начальном положении"""
        if not analysis:
            return False
        
        # Проверяем углы в локтях (должны быть близки к 180°)
        left_elbow_ok = analysis.get('left_elbow_angle', 180) >= self.config.initial_elbow_angle_min
        right_elbow_ok = analysis.get('right_elbow_angle', 180) >= self.config.initial_elbow_angle_min
        
        # Проверяем углы в коленях (ноги должны быть прямыми)
        left_knee_ok = analysis.get('left_knee_angle', 180) >= self.config.knee_angle_min
        right_knee_ok = analysis.get('right_knee_angle', 180) >= self.config.knee_angle_min
        
        # Проверяем угол таза (тело должно быть прямым)
        hip_ok = analysis.get('hip_angle', 180) >= self.config.hip_angle_min
        
        return left_elbow_ok and right_elbow_ok and left_knee_ok and right_knee_ok and hip_ok
    
    def is_final_position(self, analysis: dict) -> bool:
        """Проверяет, находится ли человек в финальном положении"""
        if not analysis:
            return False
        
        # Проверяем углы в локтях (должны быть меньше 90°)
        left_elbow_ok = analysis.get('left_elbow_angle', 180) <= self.config.final_elbow_angle_max
        right_elbow_ok = analysis.get('right_elbow_angle', 180) <= self.config.final_elbow_angle_max
        
        # Проверяем, что подбородок над плечами
        chin_ok = analysis.get('chin_above_shoulders', False)
        
        # Проверяем углы в коленях (ноги должны оставаться прямыми)
        left_knee_ok = analysis.get('left_knee_angle', 180) >= self.config.knee_angle_min
        right_knee_ok = analysis.get('right_knee_angle', 180) >= self.config.knee_angle_min
        
        # Проверяем угол таза (тело должно оставаться прямым)
        hip_ok = analysis.get('hip_angle', 180) >= self.config.hip_angle_min
        
        return left_elbow_ok and right_elbow_ok and chin_ok and left_knee_ok and right_knee_ok and hip_ok
    
    def update_state(self, analysis: dict):
        """Обновляет состояние на основе анализа позы"""
        current_time = time.time()
        
        if self.state == PullupState.INITIAL:
            if self.is_final_position(analysis):
                self.state = PullupState.FINAL
                self.hold_start_time = current_time
            elif not self.is_initial_position(analysis):
                self.state = PullupState.PULLING
        
        elif self.state == PullupState.PULLING:
            if self.is_final_position(analysis):
                self.state = PullupState.FINAL
                self.hold_start_time = current_time
            elif self.is_initial_position(analysis):
                self.state = PullupState.INITIAL
        
        elif self.state == PullupState.FINAL:
            if self.is_final_position(analysis):
                # Проверяем, удерживается ли положение достаточно долго
                if self.hold_start_time and (current_time - self.hold_start_time) >= self.config.hold_duration:
                    self.state = PullupState.LOWERING
            else:
                # Если положение не удерживается, возвращаемся к подтягиванию
                self.state = PullupState.PULLING
                self.hold_start_time = None
        
        elif self.state == PullupState.LOWERING:
            if self.is_initial_position(analysis):
                # Завершили полный цикл
                self.rep_count += 1
                self.play_rep_sound()
                self.state = PullupState.INITIAL
                self.last_rep_time = current_time
            elif self.is_final_position(analysis):
                # Вернулись в финальное положение
                self.state = PullupState.FINAL
                self.hold_start_time = current_time
    
    def draw_skeleton(self, frame: np.ndarray, landmarks):
        """Отрисовывает скелет на кадре"""
        if not landmarks or len(landmarks) < 17:
            return
        
        # COCO connections для отрисовки скелета
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # голова и руки
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # плечи и руки
            (5, 11), (6, 12), (11, 12),  # туловище
            (11, 13), (13, 15), (12, 14), (14, 16)  # ноги
        ]
        
        # Рисуем ключевые точки
        for i, point in enumerate(landmarks):
            if i < 17 and point[2] > self.config.confidence_threshold:
                x, y = int(point[0]), int(point[1])
                # Разные цвета для разных частей тела
                if i in [0, 1, 2, 3, 4]:  # голова
                    color = (255, 0, 0)  # красный
                elif i in [5, 6, 7, 8, 9, 10]:  # руки
                    color = (0, 255, 0)  # зеленый
                elif i in [11, 12]:  # таз
                    color = (0, 0, 255)  # синий
                else:  # ноги
                    color = (255, 255, 0)  # желтый
                
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)
        
        # Рисуем соединения
        for connection in connections:
            if (connection[0] < len(landmarks) and connection[1] < len(landmarks) and
                landmarks[connection[0]][2] > self.config.confidence_threshold and
                landmarks[connection[1]][2] > self.config.confidence_threshold):
                
                pt1 = (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1]))
                pt2 = (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 3)
    
    def draw_info(self, frame: np.ndarray, analysis: dict):
        """Отрисовывает информацию на кадре"""
        h, w = frame.shape[:2]
        
        # Фон для текста
        cv2.rectangle(frame, (10, 10), (450, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 220), (255, 255, 255), 2)
        
        # Счетчик повторений
        cv2.putText(frame, f"Reps: {self.rep_count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Текущее состояние
        state_colors = {
            PullupState.INITIAL: (0, 255, 0),    # Зеленый
            PullupState.PULLING: (0, 255, 255),  # Желтый
            PullupState.FINAL: (0, 165, 255),    # Оранжевый
            PullupState.LOWERING: (255, 0, 0)    # Красный
        }
        state_color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"State: {self.state.value}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Информация об углах
        if 'left_elbow_angle' in analysis:
            cv2.putText(frame, f"L Elbow: {analysis['left_elbow_angle']:.1f}°", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if 'right_elbow_angle' in analysis:
            cv2.putText(frame, f"R Elbow: {analysis['right_elbow_angle']:.1f}°", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if 'left_knee_angle' in analysis:
            cv2.putText(frame, f"L Knee: {analysis['left_knee_angle']:.1f}°", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if 'right_knee_angle' in analysis:
            cv2.putText(frame, f"R Knee: {analysis['right_knee_angle']:.1f}°", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Информация об угле таза
        if 'hip_angle' in analysis:
            hip_color = (0, 255, 0) if analysis['hip_angle'] >= self.config.hip_angle_min else (0, 0, 255)
            cv2.putText(frame, f"Hip Angle: {analysis['hip_angle']:.1f}°", (20, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, hip_color, 1)
        
        # Индикатор подбородка
        if 'chin_above_shoulders' in analysis:
            chin_status = "Chin Above" if analysis['chin_above_shoulders'] else "Chin Below"
            chin_color = (0, 255, 0) if analysis['chin_above_shoulders'] else (0, 0, 255)
            cv2.putText(frame, chin_status, (20, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, chin_color, 1)
        
        # Инструкции
        instructions = [
            "Instructions:",
            "1. Stand under the bar",
            "2. Grab the bar",
            "3. Straighten arms and legs",
            "4. Keep body straight",
            "5. Pull up to chin over bar",
            "6. Lower to starting position"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 180 + i * 20
            cv2.putText(frame, instruction, (w - 400, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обрабатывает кадр и возвращает его с нарисованной информацией"""
        # Обрабатываем позу с помощью rtmlib
        results = self.pose_tracker(frame)
        # Получаем ключевые точки
        landmarks = results[0]
        # Анализируем позу
        analysis = self.analyze_pose(landmarks)
        
        # Обновляем состояние
        self.update_state(analysis)
        
        # Рисуем скелет
        if landmarks is not None:
            print('11')
            self.draw_skeleton(frame, landmarks)
        
        # Рисуем информацию
        self.draw_info(frame, analysis)
        
        return frame
    
    def run(self):
        """Запускает основной цикл обработки видео"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Failed to open camera")
            return
        
        print("Starting pullup counter (rtmlib)...")
        print("Press 'q' to exit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")

                # Обрабатываем кадр
                processed_frame = self.process_frame(frame)
                
                # Показываем результат
                cv2.imshow('Pullup Counter (rtmlib)', processed_frame)
                
                # Проверяем нажатие клавиши
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"Finished. Total reps completed: {self.rep_count}")

def main():
    """Главная функция"""
    config = PullupConfig()
    counter = PullupCounter(config)
    counter.run()

if __name__ == "__main__":
    main() 