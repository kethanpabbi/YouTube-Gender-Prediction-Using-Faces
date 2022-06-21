from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import cvlib as cv
import os
from moviepy.editor import *
import re
                   
def gender_predict(video, model):
    
    clear_frames()
    
    classes = ['man','woman']

    male_fps = 0
    female_fps = 0
    total_fps = 0
    global duration
    global fps
    global frame_count

    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps
        print(duration, fps, frame_count)
        # loop through frames
        while True:
            
            total_fps += 1
            # read frame from video 
            status, frame = video.read()
            #fps = video.get(cv2.CAP_PROP_FPS)

            # apply face detection
            face, confidence = cv.detect_face(frame)
            
            # loop through detected faces
            for idx, f in enumerate(face):

                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # draw rectangle over face
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

                # crop the detected face region
                face_crop = np.copy(frame[startY:endY,startX:endX])

                if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                    continue

                # preprocessing for gender detection model
                face_crop = cv2.resize(face_crop, (250,250))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                # apply gender detection on face
                conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

                # get label with max accuracy
                idx = np.argmax(conf)
                label = classes[idx]

                text = "{}: {:.2f}%".format(label, conf[idx] * 100)

                if label == 'man':
                    male_fps += 1

                if label == 'woman':
                    female_fps += 1
                
                Y = startY - 10 if startY - 10 > 10 else startY + 10

                # write label and confidence above face rectangle
                
                cv2.putText(frame, text, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (200, 200, 200), 2)
            
            # display output
            cv2.imwrite('/Users/kethanpabbi/Desktop/Thesis/22-06/video_frames/'+str(total_fps)+'.jpg',frame)

    except: Exception
    non_human_fps = 0
    if male_fps + female_fps <= total_fps:
        non_human_fps = total_fps - (male_fps + female_fps)
        return print(f'Total Duration: {duration}\nMale Screen Time: {male_fps/fps}\tFemale Screen Time: {female_fps/fps}\tTotal Non-Human Time: {non_human_fps/fps}')

def img_to_vid():
    img_array = []

    for filename in os.listdir('/Users/kethanpabbi/Desktop/Thesis/22-06/video_frames'):
        img_array.append(os.path.join('/Users/kethanpabbi/Desktop/Thesis/22-06/video_frames',filename))
    
    img_array.sort(key=natural_keys)
    clip = ImageSequenceClip(img_array, fps=fps) 
    clip.write_videofile("/Users/kethanpabbi/Desktop/Thesis/22-06/video/video.mp4", fps=fps)
    clear_frames()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def clear_frames():
    dir = '/Users/kethanpabbi/Desktop/Thesis/22-06/video_frames'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))