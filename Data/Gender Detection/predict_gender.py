# Import Libraries
import cv2
import re
import numpy as np
import os
from moviepy.editor import *
import youtube_dl
import pandas as pd
from openpyxl import load_workbook

# The gender model architecture
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = 'weights/deploy_gender.prototxt'

# The gender model pre-trained weights
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = 'weights/gender_net.caffemodel'

# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Represent the gender classes
GENDER_LIST = ['Male', 'Female']

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"

# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Initialize frame size
frame_width = 1280
frame_height = 720


def get_faces(frame, confidence_threshold=0.5):
    '''Detect all the faces present in a frame'''
   
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
  
    # set the image as input to the NN
    face_net.setInput(blob)
   
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
   
    # initialize the result list
    faces = []
    
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
           
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces

def get_optimal_font_scale(text, width):
    '''Determine the optimal font scale based on the hosting frame width'''

    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1

# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def frame_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    '''Resize frame dimensions for faster processing'''
    # initialize the dimensions of the image to be resized and
    dim = None

    # grab the image size
    (h, w) = image.shape[:2]
    
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
   
    # check to see if the width is None
    if width is None:
        # height dimension
        r = height / float(h)
        dim = (int(w * r), height)
    
    # otherwise, the height is None
    else:
        # width dimension
        r = width / float(w)
        dim = (width, int(h * r))
    
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)

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
            
            # resize the frame if needed
            if frame.shape[1] > frame_width:
               frame = frame_resize(frame, width=frame_width)
            
            # predict the faces
            faces = get_faces(frame)
            
            # Loop over the faces detected
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]

                blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
                    227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
                
                # Predict Gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                i = gender_preds[0].argmax()
                gender = GENDER_LIST[i]
                if gender == 'Male':
                    male_fps += 1
                elif gender == 'Female':
                    female_fps += 1
                
                # Confidence of the gender
                gender_confidence_score = gender_preds[0][i]
                
                # Draw the box
                label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                print(label)
                yPos = start_y - 15
                while yPos < 15:
                    yPos += 15
                
                # get the font scale for this image size
                optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
                box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                
                # Label processed image
                cv2.putText(frame, label, (start_x, yPos),
                            cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

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
        spreedsheet(male_fps, female_fps, non_human_fps)
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
        img_array.append(os.path.join('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/video_frames/',filename))
    
    # Sort the frames in order
    img_array.sort(key=natural_keys)

    # Combine to form MP4 with required fps
    clip = ImageSequenceClip(img_array, fps=fps) 
    clip.write_videofile("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/"+str(title+format)+".mp4", fps=fps)

    remove_frames()

def spreedsheet(male_fps, female_fps, non_human_fps):
    # new dataframe with same columns
    name = str(title+format)
    df = pd.DataFrame({'Title': [name],
                    'Duration': duration, 'Male Screen time': male_fps/fps,\
                    'Female Screen Time': female_fps/fps, 'Non-Human Screen Time': non_human_fps/fps})
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel('Stats.xlsx', sheet_name='Sheet1', index=False)

if __name__ == '__main__':

    global title
    global format

    #cat
    #video_url = 'https://www.youtube.com/watch?v=HECa3bAFAYkq'

    #galway girl
    #video_url = 'https://www.youtube.com/watch?v=87gWaABqGYs'

    #news
    #video_url = 'https://www.youtube.com/watch?v=0Iu4C0mT3dw'

    #video_url = 'https://www.youtube.com/watch?v=DUqqPCPll_g'
    
    #video_url = 'https://www.youtube.com/watch?v=po02mFUhRTk'

    #friends
    #video_url = 'https://www.youtube.com/watch?v=cB-DVomcEb4'

    #funny
    video_url = 'https://www.youtube.com/watch?v=dlx8TanWFys'
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

    Quality = str(input('Enter Quality: 144p/ 240p/ 360p/ 480p/ 720p: '))

    # Check if the file with the format already exists
    if os.path.exists('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/'+str(title+Quality)+'.mp4'):
        print('File already available to download!') 
    
    else:
        # url
        url = ''

        for f in formats:

            # Get right format
            if f.get('format_note',None) == Quality:

                format = Quality

                #get the video url
                url = f.get('url',None)
                break
        
        if url != '':

            gender_predict(url)
            img_to_vid()

        else:
            for f in formats:

                # Set default to 360p
                if f.get('format_note',None) == '360p':

                    print(f"Sorry {str(Quality)} is not available, here is 360p instead:")
                    format = '360p'

                    if os.path.exists('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/processed_video/'+str(title+format)+'.mp4'):
                        print('File already available to download!') 
                        break

                    else:
                        #get the video url
                        url = f.get('url',None)

                        gender_predict(url)
                        img_to_vid()
                        break
