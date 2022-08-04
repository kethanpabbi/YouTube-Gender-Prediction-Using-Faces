import cv2
import numpy as np
from retinaface import RetinaFace as rf
from model.model import ResMLP


GENDER_MODEL = 'weights/deploy_gender.prototxt'

GENDER_PROTO = 'weights/gender_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Represent the gender classes
GENDER_LIST = ['Male', 'Female']

gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def predict_age_gender():

    path = '/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/images-2.jpeg'
    img = cv2.imread(path)
    obj = rf.detect_faces(img, threshold = 0.75)
    for key in obj.keys():
        identity = obj[key]
        
        face_area = identity["facial_area"]
        cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg', cv2.resize(img[face_area[1]:face_area[3], face_area[0]:face_area[2]], (227, 227)))
        # get the font scale for this image size
        blob = cv2.dnn.blobFromImage(image=cv2.imread('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg'), scalefactor=1.0, size=(
                    227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
                
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        if gender == 'Male':
            print('male')
        elif gender == 'Female':
            print('female')
        
        # Confidence of the gender
        gender_confidence_score = gender_preds[0][i]
        
        # Draw the box
        label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
        print(label)
        
        cv2.rectangle(img, (face_area[2], face_area[3]), (face_area[0], face_area[1]), (147, 20, 255), 1)
        

    
    
    
    #embeddings = io.BytesIO(emb1)
    #embeddings = np.load(embeddings, allow_pickle=True)
    #embeddings = emb1.reshape(-1, 512).astype(np.float32)




    return ''


if __name__ == "__main__":

    g = predict_age_gender()
    print(g)