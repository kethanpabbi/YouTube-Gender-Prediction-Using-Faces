# Import Libraries
import cv2
import re
import numpy as np
import os
from moviepy.editor import *
import youtube_dl
import pandas as pd
from retinaface import RetinaFace as rf
import cvlib as cv
from deepface import DeepFace as dp
import numpy as np
import time


start_time = time.time()

def gender_predict(video):
    '''Predict the gender in a video'''

    male_fps = 0
    female_fps = 0
    total_fps = 0
    global duration
    global fps
    global frame_count

    try:
        # create a new cam object
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        print(f'Duration:{duration:.3f}, FPS:{int(fps)}, Total Frames:{frame_count}')
        
        while True:
            
            # total frames
            total_fps += 1
            
            # read frame from video 
            status, frame = cap.read()
            
            # predict the faces
            obj = rf.detect_faces(frame, threshold = 0.75)
            if type(obj) != dict:
                cv2.imshow("Gender Estimator", frame)
                cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+str(total_fps)+'.jpg',frame)
            else:
                # Loop over the faces detected
                for key in obj.keys():
                    identity = obj[key]
                    
                    face_area = identity["facial_area"]
                    cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg', frame[face_area[1]:face_area[3], face_area[0]:face_area[2]])
                    dpa = dp.analyze('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/0.jpg', actions = ["gender"],enforce_detection=False, detector_backend = 'dlib')
                    label = dpa['gender']

                    box_color = (255, 0, 0) if label == "Man" else (147, 20, 255)
                    
                    
                    # label = dpa['gender']
                    # get the font scale for this image size
                    cv2.rectangle(frame, (face_area[2], face_area[3]), (face_area[0], face_area[1]), box_color, 1)
                    
                    # Label processed image
                    cv2.putText(frame, label, (face_area[0],face_area[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                

                # Display processed image
                cv2.imshow("Gender Estimator", frame)
                cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/'+str(total_fps)+'.jpg',frame)
                
                
            
            # Quit midway
            if cv2.waitKey(1) == ord("q"):
                remove_frames()
                break
        
        # Cleanup
        cv2.destroyAllWindows()
    except: Exception
    non_human_fps = 0
    if male_fps + female_fps <= total_fps:
        non_human_fps = total_fps - (male_fps + female_fps)
        return print(f'Total Duration: {duration:.3f}\nMale Screen Time: {male_fps/fps:.3f}\nFemale Screen Time:{female_fps/fps:.3f}\nTotal Non-Human Time: {non_human_fps/fps:.3f}')

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
    clip = ImageSequenceClip(img_array, fps=fps) 
    clip.write_videofile("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/"+str(title+quality)+"RetinaFaceMtcnn.mp4", fps=int(fps))

    remove_frames()

if __name__ == '__main__':

    global title
    global quality

    #cat
    #video_url = 'https://www.youtube.com/watch?v=HECa3bAFAYkq'

    #galway girl
    #video_url = 'https://www.youtube.com/watch?v=87gWaABqGYs'

    #news
    #video_url = 'https://www.youtube.com/watch?v=0Iu4C0mT3dw'

    #video_url = 'https://www.youtube.com/watch?v=DUqqPCPll_g'
    
    #video_url = 'https://www.youtube.com/watch?v=BzLO2OKt3OU'

    #friends
    #video_url = 'https://www.youtube.com/watch?v=cB-DVomcEb4'

    #funny
    #video_url = 'https://www.youtube.com/watch?v=dlx8TanWFys'

    #ancor
    #video_url = 'https://www.youtube.com/watch?v=SHP-QWXUYoQ'
    video_url = 'https://www.youtube.com/watch?v=BzLO2OKt3OU'
    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)

    # video title
    title = alpha_num(info_dict['title'])
    print(title)

    # get video formats available
    formats = info_dict.get('formats',None)

    quality = str(input('Enter Quality: 144p/ 240p/ 360p/ 480p/ 720p: '))
    
    # Check if the file with the format already exists
    if os.path.exists('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/'+str(title+quality)+'RetinaFaceMtcnn.mp4'):
        print('File already available to download!') 
    
    else:
        path = '/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4'
        gender_predict(path)
        img_to_vid()

    end_time = time.time()
    print(f"Execution time: {end_time-start_time}")  #86 mins for a min long video
