from fastapi import FastAPI, WebSocket
import numpy as np
import cv2
from rtmlib import Body

app = FastAPI()
# 1. Раздача статических файлов
wholebody = Body(mode='lightweight', backend='onnxruntime', device='cpu')
from .lib import calculate_angle_by_keypoints, detected_position, PositionEnum  # твои функции

@app.websocket("/ws/inference")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pushup_count = 0
    state = PositionEnum.UNKNOWN

    while True:
        data = await websocket.receive_text()
        frames = eval(data)  # безопаснее использовать json.loads(data)
        buf1 = np.asarray(frames["cam1"], dtype=np.uint8)
        buf2 = np.asarray(frames["cam2"], dtype=np.uint8)

        frame1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)
        frame2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

        # можно выбирать одну из камер, например frame1
        keypoints, scores = wholebody(frame1)
        kp, sc = keypoints[0], scores[0]

        angle_elbow = calculate_angle_by_keypoints(sc, kp, 5, 7, 9)
        angle_hip = calculate_angle_by_keypoints(sc, kp, 5, 11, 13)
        pos = detected_position(angle_elbow, angle_hip)

        if state == PositionEnum.UP and pos == PositionEnum.DOWN:
            state = PositionEnum.DOWN
        elif state == PositionEnum.DOWN and pos == PositionEnum.UP:
            pushup_count += 1
            state = PositionEnum.UP
        elif pos == PositionEnum.UP:
            state = PositionEnum.UP

        await websocket.send_json({
            "pushups": pushup_count,
            "position": pos.name,
            "skeleton": kp.tolist(),
            "scores": sc.tolist()
        })
