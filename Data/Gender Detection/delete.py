import os
dir = '/Users/kethanpabbi/Desktop/Thesis/Proj1/video_frames/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))