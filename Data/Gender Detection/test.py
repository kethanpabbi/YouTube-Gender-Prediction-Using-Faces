from deepface import DeepFace

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
analysis = DeepFace.analyze(img_path = "/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Facial-attribute-analysis-with-deep-learning-using-the-deep-face-library copy.jpg", actions = ["gender"])
print(analysis)