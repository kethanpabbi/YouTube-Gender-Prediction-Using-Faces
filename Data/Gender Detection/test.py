import cv2
import re
import numpy as np
import os
from moviepy.editor import *
from sympy import fps
import youtube_dl
import pandas as pd
from retinaface import RetinaFace as rf
import cvlib as cv
from deepface import DeepFace as dp
import numpy as np
import time

def alpha_num(text):
    return str(re.sub(r'[^A-Za-z0-9 ]+', '', text))

def atoi(text):
    '''Return numerical values'''
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''Return list of values'''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def remove_frames():
    '''Delete frames'''
    # Delete frames
    dir = '/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
def img_to_vid():
    '''Convert sequence of frames into MP4 video'''

    img_array = []

    # Make an array of frames to be combined
    for filename in os.listdir('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'):
        if filename != '0.jpg':
            img_array.append(os.path.join('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/',filename))
    
    # Sort the frames in order
    img_array.sort(key=natural_keys)

    # Combine to form MP4 with required fps
    clip = ImageSequenceClip(img_array, fps=23) 
    clip.write_videofile("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/"+str(title+quality)+"Face_detect_FacenetDlib.mp4", fps=int(fp))

    remove_frames()

global title
global quality
global fp
fp = 25
title = 'Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021'
quality = '360p'
img_to_vid()