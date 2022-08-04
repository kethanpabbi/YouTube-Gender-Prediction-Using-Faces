from arcface import ArcFace
import cv2
import numpy as np
from sklearn.feature_extraction import img_to_graph
import torch
from retinaface import RetinaFace as rf
from model.model import ResMLP
from utils import enable_dropout, forward_mc, read_json

def predict_age_gender():

    face_rec = ArcFace.ArcFace()
    models = {"gender": None}
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    for model_ in ["gender"]:
        model = ResMLP(**read_json(f"./models/{model_}.json")["arch"]["args"])
        checkpoint = f"models/{model_}.pth"
        checkpoint = torch.load(checkpoint, map_location=torch.device(device))
        
        state_dict = checkpoint["state_dict"]
       
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        models[model_] = model
    path = '/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/images-2.jpeg'
    img = cv2.imread(path)
    obj = rf.detect_faces(img, threshold = 0.75)
    for key in obj.keys():
        identity = obj[key]
        
        face_area = identity["facial_area"]
        cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg', cv2.resize(img[face_area[1]:face_area[3], face_area[0]:face_area[2]], (112, 112)))
        # get the font scale for this image size
        emb1 = face_rec.calc_emb('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg')
        embedding = emb1.reshape(1, 512)
        gender_mean, gender_entropy = forward_mc(models["gender"], embedding)
        print(forward_mc(models["gender"], embedding))
        gender = {"m": 1 - gender_mean, "f": gender_mean, "entropy": gender_entropy}
        print(gender)

        
        cv2.rectangle(img, (face_area[2], face_area[3]), (face_area[0], face_area[1]), (147, 20, 255), 1)
        

    
    
    
    #embeddings = io.BytesIO(emb1)
    #embeddings = np.load(embeddings, allow_pickle=True)
    #embeddings = emb1.reshape(-1, 512).astype(np.float32)


    dict(sorted(gender.items(), key=lambda item: item[1]))

    return list(gender.keys())[0]


if __name__ == "__main__":

    g = predict_age_gender()
    print(g)