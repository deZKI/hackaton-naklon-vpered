import cv2

index = 0
while True:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        print(f"Камера найдена: {index}")
        cap.release()
    index += 1
