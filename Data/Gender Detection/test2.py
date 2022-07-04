import cv2
#from test import *
import mediapipe as mp
import time

cap = cv2.VideoCapture('/Users/kethanpabbi/Downloads/20180402-114759/Pexels Videos 2796078.mp4')
ptime = 0

mpface = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils
face = mpface.FaceDetection(0.75)

while True:
    success, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face.process(frameRGB)
    print(res)
    if res.detections:
        for id, detection in enumerate(res.detections):
            bbox = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            box = int(bbox.xmin*iw), int(bbox.ymin*ih),\
                int(bbox.width*iw), int(bbox.height*ih)
            cv2.rectangle(frame, box, (255,0,255), 2)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime 
    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0), 2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(30)