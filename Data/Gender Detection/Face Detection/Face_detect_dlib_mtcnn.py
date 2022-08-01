# Import Libraries
import cv2
import numpy as np
from mtcnn import MTCNN
import dlib
import torch
import time

start_time = time.time()

def gender_predict():
    '''Predict the gender in a video'''
    
    global count
    total_fps = 0
    count = 0
    face_detect = dlib.cnn_face_detection_model_v1("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/weights/mmod_human_face_detector.dat")
    detector = MTCNN()

    # create a new cam object
    cap = cv2.VideoCapture("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4")
    a,b = [],[]
    try:
        while True:
            total_fps += 1
            status, frame = cap.read()
            frame = cv2.resize(frame, (600, 400))
            
            # predict the faces
            faces = face_detect(frame, 1)
            boxes = detector.detect_faces(frame)
            for box in boxes:
                count += 1
                x, y, w, h = box['box']

                cv2.rectangle(frame, (x, y), (x+w, y+h), 
                                    (255, 0, 255), 1)
                    
            for face in faces:
                # In dlib in order to extract points we need to do this
                x1 = face.rect.left()
                y1 = face.rect.bottom()
                x2 = face.rect.right()
                y2 = face.rect.top()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            
            # Display processed image
            cv2.imshow("Gender Estimator", frame)
            cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+str(total_fps)+'.jpg',frame)
            # Quit midway
            if cv2.waitKey(1) == ord("q"):
                break
    except: Exception
    print(count)

if __name__ == '__main__':
    gender_predict()
    end_time = time.time()
    print(f"Execution time: {end_time-start_time}") 

