import cv2
from tensorflow.keras.models import load_model
import numpy as np
import youtube_dl
from video import *

if __name__ == '__main__':

    # load model
    #model = load_model('/Users/kethanpabbi/Downloads/Gender-Detection-master/gender_detection.model')
    model = load_model('/Users/kethanpabbi/Downloads/Gender-Detection-master/weights.h5')
    
    #cat
    #video_url = 'https://www.youtube.com/watch?v=HECa3bAFAYkq'

    #galway girl
    #video_url = 'https://www.youtube.com/watch?v=87gWaABqGYs'

    #news
    #video_url = 'https://www.youtube.com/watch?v=0Iu4C0mT3dw'

    #video_url = 'https://www.youtube.com/watch?v=DUqqPCPll_g'
    
    video_url = 'https://www.youtube.com/watch?v=po02mFUhRTk'
    ydl_opts = {}

    # create youtube-dl object
    ydl = youtube_dl.YoutubeDL(ydl_opts)

    # set video url, extract video information
    info_dict = ydl.extract_info(video_url, download=False)

    # get video formats available
    formats = info_dict.get('formats',None)

    Quality = str(input('Enter Quality: 144p/ 240p/ 360p/ 480p/ 720p: '))

    # url
    url = ''

    for f in formats:

        # I want the lowest resolution, so I set resolution as 144p
        if f.get('format_note',None) == Quality:

            #get the video url
            url = f.get('url',None)
            break
    
    if url != '':
        # open url with opencv
        cap = cv2.VideoCapture(url)

        gender_predict(cap, model)
        img_to_vid()  
        cap.release()

    else:
        for f in formats:
            # I want the lowest resolution, so I set resolution as 144p
            if f.get('format_note',None) == '360p':

                print(f"Sorry {str(Quality)} is not available, here is 360p instead:")

                #get the video url
                url = f.get('url',None)

                # open url with opencv
                cap = cv2.VideoCapture(url)

                gender_predict(cap, model)
                img_to_vid()
                cap.release()
                break
    
    