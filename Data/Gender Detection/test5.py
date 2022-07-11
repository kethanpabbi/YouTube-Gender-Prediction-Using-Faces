from retinaface import RetinaFace as rf
import cv2
import cvlib as cv
from deepface import DeepFace as dp
import numpy as np


total_fps = 0
cap = cv2.VideoCapture('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Joey turns 30 (Friends).mp4')

# path = 'img1.jpg'
# img = cv2.imread(path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps
print(f'Duration:{duration:.3f}, FPS:{int(fps)}, Total Frames:{frame_count}')

while True:
    total_fps += 1
    success, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    obj = rf.detect_faces(frame)
    for key in obj.keys():
        identity = obj[key]

        face_area = identity["facial_area"]
        cv2.rectangle(frame, (face_area[2], face_area[3]), (face_area[0], face_area[1]), (255, 255, 255), 1)
        # crop = cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+'0.jpg',frame[(face_area[2], face_area[3]), (face_area[0], face_area[1])])
        # dpa = dp.analyze(crop, actions = ["gender"], detector_backend = 'ssd')
        # print(dpa)
        # label = dpa['gender']
        # cv2.putText(frame, label, (face_area[2],face_area[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+str(total_fps)+'.jpg',frame)
    #cv2.imshow("Gender Estimator", frameRGB)
    

# obj1 = dp.analyze('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/img2.jpg',actions = ["gender"], detector_backend = 'retinaface')
# print(obj1)