from retinaface import RetinaFace as rf
import cv2
import time

start_time = time.time()
 
cap = cv2.VideoCapture("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4")
count, face_count = 0, 0
try:
    while True:
        # read frame from video 
        status, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))
        # predict the faces
        obj = rf.detect_faces(frame, threshold = 0.75)
        if type(obj) != dict:
            continue
        else:
            count += 1
            # Loop over the faces detected
            for key in obj.keys():
                face_count += 1
                identity = obj[key]
                
                face_area = identity["facial_area"]
                cv2.rectangle(frame, (face_area[2], face_area[3]), (face_area[0], face_area[1]), (255, 255, 0), 1)
            

            # Display processed image
            cv2.imshow("Gender Estimator", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
except: Exception
print(count)
print(face_count)
cap.release()
cv2.destroyAllWindows()
end_time = time.time()
print(f"Execution time: {end_time-start_time}") 