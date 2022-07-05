import cv2

def face_video():
    face_cascade = cv2.CascadeClassifier('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/haarcascade_mcs_nose.xml')
    video = cv2.VideoCapture('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Joey turns 30 (Friends).mp4')
    try:
        while True:
            ret, image = video.read()
            
            if not ret:
                break
            image = cv2.flip(image, 1)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=10)
            #print(faces)
            for(x, y, w, h) in faces:
                cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 1)
                
            cv2.imshow('Face Detector', image)
            k=cv2.waitKey(10)
            
            if k==ord('q'):
                break

        video.release()
    except: Exception
    cv2.destroyAllWindows()



        
if __name__ == '__main__':
    face_video()
