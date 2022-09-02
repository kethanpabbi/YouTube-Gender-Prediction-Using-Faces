import dlib
import cv2
import time

start_time = time.time()
#We create the model here with the weights placed as parameters
face_detect = dlib.cnn_face_detection_model_v1("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/weights/mmod_human_face_detector.dat")

cap = cv2.VideoCapture("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4")
count = 0
try:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))
    
        faces = face_detect(frame, 1)
            
        for face in faces:
            count += 1
            # In dlib in order to extract points we need to do this
            x1 = face.rect.left()
            y1 = face.rect.bottom()
            x2 = face.rect.right()
            y2 = face.rect.top()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
except: Exception
   
print(count)
cap.release()
cv2.destroyAllWindows()
end_time = time.time()
print(f"Execution time: {end_time-start_time}") 