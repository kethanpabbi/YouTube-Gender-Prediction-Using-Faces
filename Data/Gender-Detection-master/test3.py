from pytube import YouTube
from tensorflow.keras.models import load_model
import cv2
from video import *

link="https://www.youtube.com/watch?v=87gWaABqGYs"

# load model
model = load_model('gender_detection.model')

try: 
    # object creation using YouTube
    # which was imported in the beginning 
    yt = YouTube(link) 
except: 
    print("Connection Error") #to handle exception 
  
# filters out all the files with "mp4" extension 

# mp4files = yt.filter('mp4') 
  
# get the video with the extension and
# resolution passed in the get() function 
d_video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1]
try: 
    video = cv2.VideoCapture(d_video)
    male_time, female_time, duration = gender_predict(video, model)

    # release resources
    video.release()
    cv2.destroyAllWindows()

    if male_time != None:
        non_human_duration = 0
        if male_time + female_time <= duration:
            non_human_duration = duration - (male_time + female_time)
        print(f'Total Duration: {duration}\nMale Screen Time: {male_time}\tFemale Screen Time: {female_time}\tTotal Non-Human Time: {non_human_duration}')
except: 
    print("Some Error!") 
print('Task Completed!') 