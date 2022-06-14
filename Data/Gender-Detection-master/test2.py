from video import *
from tensorflow.keras.models import load_model

# load model
model = load_model('gender_detection.model')

# video
#video = cv2.VideoCapture('/Users/kethanpabbi/Downloads/Gender-Detection-master/pexels-tima-miroshnichenko-5384347.mp4')
#video = cv2.VideoCapture('/Users/kethanpabbi/Downloads/Gender-Detection-master/3260327807.mp4')
# animals
#video = cv2.VideoCapture('/Users/kethanpabbi/Downloads/Gender-Detection-master/production ID_3987730.mp4')
video = cv2.VideoCapture('/Users/kethanpabbi/Downloads/Gender-Detection-master/Pexels Videos 2796078.mp4')

male_time, female_time, duration = gender_predict(video, model)

# release resources
video.release()
cv2.destroyAllWindows()

if male_time != None:
    non_human_duration = 0
    if male_time + female_time <= duration:
        non_human_duration = duration - (male_time + female_time)
    print(f'Total Duration: {duration}\nMale Screen Time: {male_time}\tFemale Screen Time: {female_time}\tTotal Non-Human Time: {non_human_duration}')