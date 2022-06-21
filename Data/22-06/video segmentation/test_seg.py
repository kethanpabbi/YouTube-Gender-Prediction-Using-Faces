import pixellib
import cv2
from pixellib.instance import custom_segmentation

video = "/Users/kethanpabbi/Desktop/Thesis/trial/Pexels Videos 2796078.mp4"
test_video = custom_segmentation()
test_video.inferConfig(num_classes=  2, class_names=["BG", "male", "female"])
#test_video.load_model("pointrend_resnet50.pkl")
test_video.load_model("/Users/kethanpabbi/Desktop/Thesis/trial/weights.h5")
test_video.process_video(video, show_bboxes = True,  output_video_name="video_out2.mp4", frames_per_second=25)

# fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
# frame_count = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
# print(fps)
# print(frame_count)