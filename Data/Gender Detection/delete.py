import os
dir = '/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))