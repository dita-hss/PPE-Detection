from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

capture = cv2.VideoCapture('media/ppe-2.mp4')

model = YOLO('best.pt')

classNames = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
prev_frame_time = 0
new_frame_time = 0
red_color = [0,0,255]
green_color = [0,255,0]
while True:
    new_frame_time = time.time()
    success, img = capture.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            #confidennce
            conf = math.ceil((box.conf[0] * 100)) / 100

            #class name
            cls = int(box.cls[0])
            current_class = classNames[cls]

            if current_class == 'no-helmet' or current_class == 'no-vest':
                cv2.rectangle(img, (x1, y1) , (x2, y2), red_color, 3)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, 
                               colorB=red_color, colorT=[0,0,0], colorR=red_color)
            else: 
                cv2.rectangle(img, (x1, y1) , (x2, y2), green_color, 3)
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1, 
                               colorB=green_color, colorT=[0,0,0], colorR=green_color)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.imshow("Image", img)
    cv2.waitKey(1)
