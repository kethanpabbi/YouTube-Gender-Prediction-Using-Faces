import numpy as np
import cv2 

vidcap = cv2.VideoCapture('/Users/kethanpabbi/Downloads/Gender-Detection-master/pexels-tima-miroshnichenko-5384347.mp4')
assert vidcap.isOpened()

fps_in = vidcap.get(cv2.CAP_PROP_FPS)
print(fps_in)
index_in = -1
index_out = -1

while True:
    outcome = vidcap.read()
    if not outcome: break
    index_in += 1

    out_due = int(index_in / fps_in)
    print(out_due)
    if out_due > index_out:
        outcome, frame = vidcap.read()
        if not outcome: break
        index_out += 1
print(index_out)