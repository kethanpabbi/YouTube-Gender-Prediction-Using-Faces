from retinaface import RetinaFace as rf
import cv2
from deepface import DeepFace as dp

path = 'img1.jpg'
img = cv2.imread(path)
obj = rf.detect_faces(path)

for key in obj.keys():
    identity = obj[key]

    face_area = identity["facial_area"]
    cv2.rectangle(img, (face_area[2], face_area[3]), (face_area[0], face_area[1]), (255, 255, 255), 1)
#cv2.imshow("Gender Estimator", img)
#cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+'img2.jpg',img)


obj = dp.analyze('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/img2.jpg', detector_backend = 'retinaface')

print(obj)