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
from keras.utils.vis_utils import plot_model

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


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

global camera
global frame
predsHistory = [collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5), collections.deque(maxlen=5)]


# We might need to display predictions for multiple faces. We will distinguish each by 
# by individual colour values. Colours MUST BE EASILY DISTINGUISHABLE
# Create an array of colour values
# Source: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colours = [
	(75, 180, 60), # Green
	(200, 0, 130), # Blue
	(48, 130, 245), # Orange
	(180, 30, 145), # Purple
	(240, 240, 70), # Cyan
	(25, 255, 225), # Yellow	
	(230, 50, 240), # Magenta
	(60, 245, 210), # Lime
	(128, 128, 0), # Teal
	(40, 110, 170), # Brown
	(0, 0, 128), # Maroon
	(195, 255, 170), # Mint
	(0, 128, 128), # Olive
	(128, 0, 0), # Navy
	(128, 128, 128), # Grey
]

# The emotion classification probablites will be displayed on a 
# black canvas that's 300x220 pixels. Create 15 of these canvases
# to work with classification of up to 15 faces
canvas_probalities = np.zeros((15, 220, 300, 3), dtype="uint8")
# Instialise display frame to be 1400x700 pixels
displayFrame = np.zeros((700, 1200, 3), dtype="uint8")



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
	
	
	# We will display the frame that we feed to the network. We display
	# this frame so that what the user sees is what is being feed to
	# the network. However we enlarge the frame to make it more visible	
	frameEnlarged = cv2.resize(frame, (0,0), fx=1.6, fy=1.6)
	# Place the enlarged frame at the top left corner of the display frame
	displayFrame[0:len(frameEnlarged),0:len(frameEnlarged[1])] = frameEnlarged
	# The currenly captured webcam image is displayed wether it has a face in it
	# or not. However the probablities of emotions, which may be for one
	# or more faces, which are shown in the bottom half of the display frame
	# must be cleared and redrawn if a face was captured and emotions probablities
	# determined
	displayFrame[430:] = 0


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

			# The final list set of predictions is the mean of the last 5 predictions
			# this smoothes out the noise and stops the prediction display from flickering
			# from one state to another. If there is  more than one face in the image,
			# disable the filtering. The system does not keep track of where a face is,
			# on a frame-by-frame basis, of multiple faces exist, the face that was
			# found first is not guaranteed to be the face that will be found first in 
			# the next frame

			if len(rects) == 1:
				predsFiltered = np.array([np.mean(predsHistory[0]), np.mean(predsHistory[1]), \
					np.mean(predsHistory[2]), np.mean(predsHistory[3]), \
					np.mean(predsHistory[4]), np.mean(predsHistory[5])])
			elif len(rects) > 1:
				predsFiltered  = np.array([predsHistory[0][4], predsHistory[1][4], \
					predsHistory[2][4], predsHistory[3][4], \
					predsHistory[4][4], predsHistory[5][4]])



			label = EMOTIONS[predsFiltered.argmax()]


			# loop over the labels + probabilities and draw them
			for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predsFiltered)):
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
			# draw the label of classified emotion, the rectangle showing
			# the ROI onto the cloned frame.
			cv2.putText(frameClone, label, (fX, fY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[number], 2)
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
				colours[number], 2)

			# Enlarge the cloned frame to make it more visible
			cloneEnlarged = cv2.resize(frameClone, (0,0), fx=1.6, fy=1.6) 	
			# Display the cloned and elarged frame in the top left corner
			# of the display frame
			displayFrame[0:len(cloneEnlarged),0:len(cloneEnlarged[1])] = cloneEnlarged
			# Display the probablities of emotions at the bottom of display
			# frame. Probabilty canvas of the current face bottom left. For
			# each new face add its canva to the right previous one in the
			# display frame
			displayFrame[430:650,(300 * (number)):(300 * (number+1))] = canvas_probalities[number]


		# # show our classifications + probabilities
		# cv2.imshow("Probabilities: Face " + str(number+1), canvas_probalities[number])
	# # show our classifications + probabilities
	cv2.imshow("Video", displayFrame)

	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()