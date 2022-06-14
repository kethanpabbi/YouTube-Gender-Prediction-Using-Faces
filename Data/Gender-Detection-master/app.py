# save this as app.py
from flask import Flask, escape, request
from video2 import *
from flask import Response
from flask import render_template
import threading
import argparse
import datetime
import imutils
from tensorflow.keras.models import load_model
import cv2

model = load_model('gender_detection.model')
video_url = 'https://www.youtube.com/watch?v=0Iu4C0mT3dw'

app = Flask(__name__)
@app.route("/")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, default='172.16.22.220',
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, default='8080',
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=gender_predict(video_url, model), args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()