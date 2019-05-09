# USAGE
# python emotion_detector.py --cascade haarcascade_frontalface_default.xml \
#	--model output/epoch_75.hdf5
#
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import urllib
import collections

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained emotion detector CNN")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-s", "--source",
	help="source of video feed [ip or serial interface]")
ap.add_argument("-a", "--address",
	help="address of camera [ip address or serial interface address]")
args = vars(ap.parse_args())

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
	"neutral"]

global camera
global frame
predsHistory = [collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5)]


# We might need to display predictions for multiple faces. We will distinguish each by 
# by individual colour values. Colours MUST BE EASILY DISTINGUISHABLE
# Create an array of colour values
# Source: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colours = [
	(60, 180, 75), # Green
	(0, 130, 200), # Blue
	(245, 130, 48), # Orange
	(145, 30, 180), # Purple
	(70, 240, 240), # Cyan
	(255, 225, 25), # Yellow	
	(240, 50, 230), # Magenta
	(210, 245, 60), # Lime
	(0, 128, 128), # Teal
	(170, 110, 40), # Brown
	(128, 0, 0), # Maroon
	(170, 255, 195), # Mint
	(128, 128, 0), # Olive
	(0, 0, 128), # Navy
	(128, 128, 128), # Grey
]

canvas_probalities = np.zeros((15, 220, 300, 3), dtype="uint8")


if not args.get("source", "ip"):
	# if a video path was not supplied, grab the reference to the webcam
	if not args.get("video", False):
			camera = cv2.VideoCapture()
			# Search the available video capture interfaces and find one.
			# Search from videocaptire interface 0 to interface 3. If a 
			# an interfaces 0 - 3 don't have valid cameras open, use -1
			# which give the last available video cap device in the array
			# of video capture devices
			for i in range (0, 4):
				if (i == 3):
					camera.open(-1)
					break
				if (camera.open(i) == True):
					break
	# otherwise, load the video
	else:
		camera = cv2.VideoCapture(args["video"])

# keep looping
while True:

	# grab the current frame from video

	# if source is IP camera gab from IP camera
	if args.get("source", "ip"):
		try:
			imgResp = urllib.request.urlopen(args.get("address"))
			imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
			frame = cv2.imdecode(imgNp, -1)
		except urllib.error.HTTPError as e:
			print('HTTPError: {}'.format(e.code))
	# else default to web cam and use OpenCV's VideoCapture() to grab 
	# current fram from web cam
	else:	
		(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		break

	# resize the frame and convert it to grayscale
	frame = imutils.resize(frame, width=300)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# initialize the canvas for the visualization, then clone
	# the frame so we can draw on it
	canvas_probalities[:] = np.zeros((220, 300, 3), dtype="uint8")
	frameClone = frame.copy()

	# detect faces in the input frame, then clone the frame so that
	# we can draw on it
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	# ensure at least one face was found before continuing
	if len(rects) > 0:
		rects = sorted(rects, reverse=True,
			key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

		# for (x, y, w, h) in rects:
		for number in range(len(rects)):
			(fX, fY, fW, fH) = rects[number]			

			# extract the face ROI from the image, then pre-process
			# it for the network
			roi = gray[fY:fY + fH, fX:fX + fW]
			roi = cv2.resize(roi, (48, 48))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# make a prediction on the ROI, then lookup the class
			# label
			preds = model.predict(roi)[0]

			# For each prediction append to the last list of last 5 predictions
			predsHistory[0].append(preds[0])
			predsHistory[1].append(preds[1])
			predsHistory[2].append(preds[2])
			predsHistory[3].append(preds[3])
			predsHistory[4].append(preds[4])
			predsHistory[5].append(preds[5])

			# The final list set of predictions is the average of the last 5 predictions
			# this smoothes out the noise and stops the prediction display from flickering
			# from one state to another 
			predsAvg = np.array([np.average(predsHistory[0]), np.average(predsHistory[1]), \
				np.average(predsHistory[2]), np.average(predsHistory[3]), \
				np.average(predsHistory[4]), np.average(predsHistory[5])])

			label = EMOTIONS[predsAvg.argmax()]


			# loop over the labels + probabilities and draw them
			for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predsAvg)):
			# construct the label text
				text = "{}: {:.2f}%".format(emotion, prob * 100)
				
				# draw the label + probability bar on the canvas
				w = int(prob * 300)

				cv2.rectangle(canvas_probalities[number], (5, (i * 35) + 5),
					(w, (i * 35) + 35), colours[number], -1)
				cv2.putText(canvas_probalities[number], text, (10, (i * 35) + 23),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45,
					(255, 255, 255), 2)

			# draw the label on the frame
			cv2.putText(frameClone, label, (fX, fY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[number], 2)
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
				colours[number], 2)

		# # show our classifications + probabilities
		cv2.imshow("Probabilities: Face " + str(number+1), canvas_probalities[number])
	# # show our classifications + probabilities
	cv2.imshow("Video", frameClone)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()