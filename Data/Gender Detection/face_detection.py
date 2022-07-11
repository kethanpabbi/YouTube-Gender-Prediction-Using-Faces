import mediapipe as mp
import cv2
import time

class FaceDetector():
    def __init__(self, minDetectionCon=0.75, model=1):
        self.minDetectionCon = minDetectionCon
        self.model = model
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(model_selection=model, min_detection_confidence=self.minDetectionCon)

    def findFaces(self, frame, draw=True):
        self.frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(self.frameRGB)
        #print(results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih),\
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([bbox, detection.score])

                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                cv2.putText(frame, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,\
                            3, (0,0,255), 2)

        return frame, bboxs
    
    def get_optimal_font_scale(self, text, width):
        '''Determine the optimal font scale based on the hosting frame width'''

        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10
        return 1
    



def main():
    cap = cv2.VideoCapture('/Users/kethanpabbi/Desktop/Thesis/YouTube-Gender-Prediction-Using-Faces/Data/Gender Detection/Joey turns 30 (Friends).mp4')
    pTime = 0
    detector = FaceDetector()
    while True:
        success, frame = cap.read()
        frame, bboxs = detector.findFaces(frame)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()