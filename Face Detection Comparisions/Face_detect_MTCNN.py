from mtcnn import MTCNN
import cv2
import time

start_time = time.time()
#Load the haarcascade file
detector = MTCNN()
 
cap = cv2.VideoCapture("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4")
count = 0
try:
    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))
        boxes = detector.detect_faces(frame)
        for box in boxes:
            count += 1
            x, y, w, h = box['box']

            cv2.rectangle(frame, (x, y), (x+w, y+h), 
                                (255, 0, 0), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
except: Exception
print(count)
cap.release()
cv2.destroyAllWindows()
end_time = time.time()
print(f"Execution time: {end_time-start_time}") 