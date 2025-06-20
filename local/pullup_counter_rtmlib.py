import cv2
import numpy as np
import time
import pygame
import os
from datetime import datetime
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from rtmlib import Wholebody

# Инициализация pygame для звука
pygame.mixer.init()

class PullupState(Enum):
    """Состояния выполнения подтягивания"""
    INITIAL = "initial"      # Начальное положение
    PULLING = "pulling"      # Подтягивание
    FINAL = "final"          # Финальное положение
    LOWERING = "lowering"    # Опускание


def draw_skeleton_safe(frame: np.ndarray, landmarks: np.ndarray, scores: np.ndarray, confidence_threshold: float = 0.6):
    """Безопасно отрисовывает скелет на кадре - все 133 точки"""
    if landmarks is None or scores is None or len(landmarks) < 17:
        return

    # Цвета для разных типов точек
    colors = [
        (255, 0, 0),    # красный
        (0, 255, 0),    # зеленый
        (0, 0, 255),    # синий
        (255, 255, 0),  # желтый
        (255, 0, 255),  # магента
        (0, 255, 255),  # циан
        (255, 128, 0),  # оранжевый
        (128, 0, 255),  # фиолетовый
    ]

    # Рисуем все доступные ключевые точки
    max_points = min(len(landmarks), len(scores), 133)  # Максимум 133 точки
    
    for i in range(max_points):
        if scores[i] > confidence_threshold:
            point = landmarks[i]
            x, y = int(point[0]), int(point[1])
            
            # Выбираем цвет в зависимости от индекса точки
            color = colors[i % len(colors)]
            
            # Специальные точки выделяем больше
            if i == 30:  # подбородок
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)  # большой магентовый круг
                cv2.putText(frame, f"30", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            elif i in [91, 94, 112, 115]:  # точки пальцев рук
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)  # циановые круги для пальцев
                cv2.putText(frame, f"{i}", (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            elif i < 17:  # основные COCO точки
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
            else:  # остальные точки
                cv2.circle(frame, (x, y), 2, color, -1)

    # Рисуем основные COCO соединения
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # голова
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # руки
        (5, 11), (6, 12), (11, 12),  # туловище
        (11, 13), (13, 15), (12, 14), (14, 16)  # ноги
    ]
    
    for connection in connections:
        if (connection[0] < len(landmarks) and connection[1] < len(landmarks) and
                connection[0] < len(scores) and connection[1] < len(scores) and
                scores[connection[0]] > confidence_threshold and
                scores[connection[1]] > confidence_threshold):
            pt1 = (int(landmarks[connection[0]][0]), int(landmarks[connection[0]][1]))
            pt2 = (int(landmarks[connection[1]][0]), int(landmarks[connection[1]][1]))
            cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

@dataclass
class PullupConfig:
    """Конфигурация для подсчета подтягиваний"""
    # Пороговые значения углов
    initial_elbow_angle_min: float = 160.0  # Минимальный угол в локтях для начального положения
    final_elbow_angle_max: float = 90.0     # Максимальный угол в локтях для финального положения
    knee_angle_min: float = 160.0           # Минимальный угол в коленях (ноги прямые)
    hip_angle_min: float = 130.0            # Минимальный угол таза (прямое положение)
    
    # Пороговые значения для уверенности в ключевых точках
    confidence_threshold: float = 0.6
    
    # Время удержания финального положения
    hold_duration: float = 0.5
    
    # Пороговые значения для определения подбородка над перекладиной
    chin_bar_distance_threshold: float = 30.0  # пиксели
    
    # Звуковые файлы
    rep_complete_sound: str = "audio/success.mp3"
    
    # Настройки записи видео
    enable_video_recording: bool = True
    video_output_dir: str = "recordings"
    video_fps: float = 30.0
    video_codec: str = "mp4v"  # или 'XVID'

class PullupCounter:
    """Класс для подсчета повторений подтягиваний с использованием rtmlib"""
    
    def __init__(self, config: PullupConfig):
        self.config = config
        
        # Инициализация rtmlib для отслеживания позы
        self.pose_tracker = Wholebody(mode='lightweight', backend='onnxruntime', device='cuda')
        
        self.state = PullupState.INITIAL
        self.rep_count = 0
        self.hold_start_time = None
        self.last_rep_time = 0
        
        # Загружаем звук
        self.load_sound()
        
        # Инициализация записи видео
        self.video_writer = None
        self.video_frame_count = 0
        self.video_start_time = None
    
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
    
    def start_video_recording(self, frame_width: int, frame_height: int):
        """Начинает запись видео"""
        if not self.config.enable_video_recording:
            return
        
        # Создаем директорию для записей если её нет
        os.makedirs(self.config.video_output_dir, exist_ok=True)
        
        # Генерируем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"pullup_session_{timestamp}.mp4"
        video_path = os.path.join(self.config.video_output_dir, video_filename)
        
        # Инициализируем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_codec)
        self.video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            self.config.video_fps, 
            (frame_width, frame_height)
        )
        
        self.video_start_time = time.time()
        self.video_frame_count = 0
        
        print(f"Started video recording: {video_path}")
    
    def stop_video_recording(self):
        """Останавливает запись видео"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
            if self.video_start_time:
                duration = time.time() - self.video_start_time
                print(f"Video recording stopped. Duration: {duration:.1f}s, Frames: {self.video_frame_count}")
                self.video_start_time = None
                self.video_frame_count = 0
    
    def write_video_frame(self, frame: np.ndarray):
        """Записывает кадр в видео файл"""
        if self.video_writer is not None and self.config.enable_video_recording:
            self.video_writer.write(frame)
            self.video_frame_count += 1
    
    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Вычисляет угол между тремя точками"""
        a, b, c = map(np.asarray, (a, b, c))
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return float(angle)

    def _detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Запускает детектор и приводит вывод к двум массивам 2-D."""
        kps, scr = self.pose_tracker(frame)
        if np.ndim(kps) == 3:
            kps, scr = kps[0], scr[0]
        return kps, scr

    def get_landmark_coords(self, landmarks, scores, landmark_id: int) -> Optional[Tuple[float, float]]:
        """Получает координаты ключевой точки из rtmlib"""
        if (landmarks is not None and scores is not None and 
            len(landmarks) > landmark_id and len(scores) > landmark_id):
            if scores[landmark_id] > self.config.confidence_threshold:  # confidence score
                point = landmarks[landmark_id]
                return (point[0], point[1])  # x, y coordinates
        return None
    
    def analyze_pose(self, landmarks: np.ndarray, scores: np.ndarray) -> dict:
        """Анализирует позу и возвращает ключевые параметры"""
        if landmarks is None or scores is None or len(landmarks) < 17 or len(scores) < 17:
            return {}
        
        # MediaPipe индексы ключевых точек
        # COCO базовые точки (0-16)
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
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
        
        # MediaPipe дополнительные точки
        CHIN = 30  # подбородок
        LEFT_INDEX_MCP = 112   # левый указательный палец (основание)
        LEFT_INDEX_TIP = 115   # левый указательный палец (кончик)
        RIGHT_INDEX_MCP = 91   # правый указательный палец (основание)
        RIGHT_INDEX_TIP = 94   # правый указательный палец (кончик)
        
        # Получаем координаты основных точек
        left_shoulder = self.get_landmark_coords(landmarks, scores, LEFT_SHOULDER)
        left_elbow = self.get_landmark_coords(landmarks, scores, LEFT_ELBOW)
        left_wrist = self.get_landmark_coords(landmarks, scores, LEFT_WRIST)
        left_hip = self.get_landmark_coords(landmarks, scores, LEFT_HIP)
        left_knee = self.get_landmark_coords(landmarks, scores, LEFT_KNEE)
        left_ankle = self.get_landmark_coords(landmarks, scores, LEFT_ANKLE)

        right_shoulder = self.get_landmark_coords(landmarks, scores, RIGHT_SHOULDER)
        right_elbow = self.get_landmark_coords(landmarks, scores, RIGHT_ELBOW)
        right_wrist = self.get_landmark_coords(landmarks, scores, RIGHT_WRIST)
        right_hip = self.get_landmark_coords(landmarks, scores, RIGHT_HIP)
        right_knee = self.get_landmark_coords(landmarks, scores, RIGHT_KNEE)
        right_ankle = self.get_landmark_coords(landmarks, scores, RIGHT_ANKLE)
        
        # Получаем точку подбородка (30)
        chin = self.get_landmark_coords(landmarks, scores, CHIN)
        
        # Получаем точки пальцев рук
        left_index_mcp = self.get_landmark_coords(landmarks, scores, LEFT_INDEX_MCP)
        left_index_tip = self.get_landmark_coords(landmarks, scores, LEFT_INDEX_TIP)
        right_index_mcp = self.get_landmark_coords(landmarks, scores, RIGHT_INDEX_MCP)
        right_index_tip = self.get_landmark_coords(landmarks, scores, RIGHT_INDEX_TIP)
        
        analysis = {}

        analysis["left_knee"] = left_knee
        analysis["left_ankle"] = left_ankle

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
        if left_shoulder and left_hip and left_knee:
            hip_angle = self.calculate_angle(
                left_shoulder,
                left_hip,
                left_knee
            )
            analysis['hip_angle'] = hip_angle
        
        # Проверяем положение подбородка относительно пальцев рук
        if chin:
            analysis['chin_position'] = chin
            
            # Собираем все доступные точки пальцев
            finger_points = []
            if left_index_mcp:
                finger_points.append(left_index_mcp)
            if left_index_tip:
                finger_points.append(left_index_tip)
            if right_index_mcp:
                finger_points.append(right_index_mcp)
            if right_index_tip:
                finger_points.append(right_index_tip)
            
            # Проверяем, что подбородок выше всех точек пальцев
            chin_above_fingers = True
            if finger_points:
                for finger_point in finger_points:
                    if chin[1] >= finger_point[1]:  # Y координата больше = ниже на экране
                        chin_above_fingers = False
                        break
                
                # Находим самую высокую точку пальцев для сравнения
                highest_finger_y = min(point[1] for point in finger_points)
                analysis['chin_finger_distance'] = chin[1] - highest_finger_y
            
            analysis['chin_above_fingers'] = chin_above_fingers
            analysis['finger_points_count'] = len(finger_points)
        
        # Определяем положение кистей рук
        if left_wrist:
            analysis['left_wrist_position'] = left_wrist
        
        if right_wrist:
            analysis['right_wrist_position'] = right_wrist
        
        # Проверяем симметричность хвата
        if left_wrist and right_wrist:
            wrist_level_diff = abs(left_wrist[1] - right_wrist[1])
            analysis['symmetric_grip'] = wrist_level_diff < 30
            analysis['wrist_level_difference'] = wrist_level_diff
        
        return analysis

    def is_correct_horizontal_position(self, left_knee, left_ankle):
        if left_knee and left_ankle:
            angle = self.calculate_angle(left_knee, left_ankle, [left_ankle[0] + 100, left_ankle[1]])
            return 0 < angle < 60

        return False

    def is_initial_position(self, analysis: dict) -> bool:
        """Проверяет, находится ли человек в начальном положении"""
        if not analysis or not self.is_correct_horizontal_position(analysis.get('left_knee'), analysis.get('left_ankle')):
            return False
        
        # Проверяем углы в локтях (должны быть близки к 180°)
        left_elbow_ok = analysis.get('left_elbow_angle', 180) >= self.config.initial_elbow_angle_min
        right_elbow_ok = analysis.get('right_elbow_angle', 180) >= self.config.initial_elbow_angle_min
        
        # Проверяем углы в коленях (ноги должны быть прямыми)
        left_knee_ok = analysis.get('left_knee_angle', 180) >= self.config.knee_angle_min
        right_knee_ok = analysis.get('right_knee_angle', 180) >= self.config.knee_angle_min
        
        # Проверяем угол таза (тело должно быть прямым)
        hip_ok = analysis.get('hip_angle', 180) >= self.config.hip_angle_min
        
        return left_elbow_ok and left_knee_ok and hip_ok
    
    def is_final_position(self, analysis: dict) -> bool:
        """Проверяет, находится ли человек в финальном положении"""
        if not analysis or not self.is_correct_horizontal_position(analysis.get('left_knee'), analysis.get('left_ankle')):
            return False
        
        # Проверяем углы в локтях (должны быть меньше 90°)
        left_elbow_ok = analysis.get('left_elbow_angle', 180) <= self.config.final_elbow_angle_max
        right_elbow_ok = analysis.get('right_elbow_angle', 180) <= self.config.final_elbow_angle_max
        
        # Проверяем, что подбородок выше пальцев рук
        chin_ok = analysis.get('chin_above_fingers', False)
        
        # Проверяем углы в коленях (ноги должны оставаться прямыми)
        left_knee_ok = analysis.get('left_knee_angle', 180) >= self.config.knee_angle_min
        right_knee_ok = analysis.get('right_knee_angle', 180) >= self.config.knee_angle_min
        
        # Проверяем угол таза (тело должно оставаться прямым)
        hip_ok = analysis.get('hip_angle', 180) >= self.config.hip_angle_min
        
        return left_elbow_ok and chin_ok and left_knee_ok and hip_ok
    
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

    
    def draw_info(self, frame: np.ndarray, analysis: dict):
        """Отрисовывает информацию на кадре"""
        h, w = frame.shape[:2]
        
        # # Фон для текста
        # cv2.rectangle(frame, (10, 10), (500, 300), (0, 0, 0), -1)
        # cv2.rectangle(frame, (10, 10), (500, 300), (255, 255, 255), 2)
        #
        # Счетчик повторений
        cv2.putText(frame, f"Reps: {self.rep_count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Текущее состояние
        state_colors = {
            PullupState.INITIAL: (0, 255, 0),    # Green
            PullupState.PULLING: (0, 255, 255),  # Yellow
            PullupState.FINAL: (0, 165, 255),    # Orange
            PullupState.LOWERING: (255, 0, 0)    # Red
        }
        state_color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"State: {self.state.value}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Статус записи видео
        if self.config.enable_video_recording:
            if self.video_writer is not None:
                recording_text = f"REC {self.video_frame_count} frames"
                cv2.putText(frame, recording_text, (20, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # Красный кружок - индикатор записи
                cv2.circle(frame, (480, 20), 8, (0, 0, 255), -1)
            else:
                cv2.putText(frame, "Video: Ready", (20, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
        if 'chin_above_fingers' in analysis:
            chin_status = "Chin Above Bar" if analysis['chin_above_fingers'] else "Chin Below Bar"
            chin_color = (0, 255, 0) if analysis['chin_above_fingers'] else (0, 0, 255)
            cv2.putText(frame, chin_status, (20, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, chin_color, 1)
        
        # Информация о симметричности хвата
        if 'symmetric_grip' in analysis:
            grip_status = "Grip OK" if analysis['symmetric_grip'] else "Uneven Grip"
            grip_color = (0, 255, 0) if analysis['symmetric_grip'] else (0, 255, 255)
            cv2.putText(frame, grip_status, (20, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, grip_color, 1)
        
        # Разница уровня кистей
        if 'wrist_level_difference' in analysis:
            cv2.putText(frame, f"Wrist Diff: {analysis['wrist_level_difference']:.1f}px", (20, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Информация о расстоянии подбородка до пальцев
        if 'chin_finger_distance' in analysis:
            distance = analysis['chin_finger_distance']
            distance_color = (0, 255, 0) if distance < 0 else (255, 0, 0)  # зеленый если подбородок выше
            cv2.putText(frame, f"Chin-Finger: {distance:.1f}px", (250, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, distance_color, 1)
        
        if 'finger_points_count' in analysis:
            cv2.putText(frame, f"Finger points: {analysis['finger_points_count']}", (250, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Визуальные индикаторы на изображении
        # Рисуем точку подбородка
        if 'chin_position' in analysis:
            chin_x, chin_y = int(analysis['chin_position'][0]), int(analysis['chin_position'][1])
            cv2.circle(frame, (chin_x, chin_y), 10, (255, 0, 255), 3)
            cv2.putText(frame, "Chin (30)", (chin_x + 15, chin_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Инструкции
        instructions = [
            "Instructions:",
            "1. Stand under the bar",
            "2. Grab the bar evenly",
            "3. Straighten arms and legs",
            "4. Keep body straight",
            "5. Pull chin above bar level",
            "6. Hold for 0.5 seconds",
            "7. Lower to starting position",
            "",
            "Controls: 'r' - start/stop recording, 'q' - quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = h - 220 + i * 20
            cv2.putText(frame, instruction, (w - 500, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Обрабатывает кадр и возвращает его с нарисованной информацией"""
        # Обрабатываем позу с помощью rtmlib
        landmarks, scores = self._detect(frame)
        # Получаем ключевые точки
        analysis = self.analyze_pose(landmarks, scores)

        # Обновляем состояние
        self.update_state(analysis)
        # Рисуем скелет
        draw_skeleton_safe(frame, landmarks, scores, self.config.confidence_threshold)
        
        # Рисуем информацию
        self.draw_info(frame, analysis)

        return frame
    
    def run(self):
        """Запускает основной цикл обработки видео"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            print("Error: Failed to open camera")
            return
        
        # Получаем размеры кадра
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("Starting pullup counter (rtmlib)...")
        print("Press 'r' to start/stop video recording")
        print("Press 'q' to exit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break

                # Обрабатываем кадр
                processed_frame = self.process_frame(frame)
                
                # Записываем кадр в видео если запись активна
                self.write_video_frame(processed_frame)
                
                # Показываем результат
                cv2.imshow('Pullup Counter (rtmlib)', processed_frame)
                
                # Проверяем нажатие клавиши
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Переключаем запись видео
                    if self.video_writer is None:
                        self.start_video_recording(frame_width, frame_height)
                    else:
                        self.stop_video_recording()
                
        finally:
            # Останавливаем запись видео если она активна
            self.stop_video_recording()
            cap.release()
            cv2.destroyAllWindows()
            print(f"Finished. Total reps completed: {self.rep_count}")
            if self.config.enable_video_recording:
                print(f"Video recordings saved in: {self.config.video_output_dir}/")

def main():
    """Главная функция"""
    config = PullupConfig()
    counter = PullupCounter(config)
    counter.run()

if __name__ == "__main__":
    main() 