# Import Libraries
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import time

start_time = time.time()
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "weights/deploy.prototxt.txt"

# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


def get_faces(frame, confidence_threshold=0.8):
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

def gender_predict():
    '''Predict the gender in a video'''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
    #Create the model
    mtcnn = MTCNN(keep_all=True, device=device)
    male_fps = 0
    female_fps = 0
    total_fps = 0
    global duration
    global fps
    global count
    global frame_count
    count = 0

    try:
        # create a new cam object
        cap = cv2.VideoCapture("/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Make It Extraordinary Albert Bartlett 10 Sec TV Ad 2021.mp4")
        a,b = [],[]
        while True:
            status, frame = cap.read()
            frame = cv2.resize(frame, (600, 400))
            
            # predict the faces
            faces = get_faces(frame)
            boxes, conf = mtcnn.detect(frame)

            # If there is no confidence that in the frame is a face, don't draw a rectangle around it
            if conf[0] !=  None:
                for (x, y, w, h) in boxes:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(frame, (x, y), (w, h), (255, 255, 0), 1)
                    a.append((((x, y, w, h))))

            # Loop over the faces detected
            for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                face_img = frame[start_y: end_y, start_x: end_x]

                blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(
                    227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)

                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
                b.append((((start_x, start_y, end_x, end_y))))
            # Display processed image
            cv2.imshow("Gender Estimator", frame)
            # Quit midway
            if cv2.waitKey(1) == ord("q"):
                break
        
        # Cleanup
        cv2.destroyAllWindows()
    except: Exception
    print(count)
    intersection = set(a).intersection(b)
    print(list(intersection))

if __name__ == '__main__':
    gender_predict()
    end_time = time.time()
    print(f"Execution time: {end_time-start_time}") 

