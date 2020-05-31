# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from threading import Thread
from imutils.video import WebcamVideoStream
#from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []

	# loop over the detections
	Belum = False
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			'''
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			'''
			if(Belum == False):
				locs.append((startX, startY, endX, endY))
				Belum = True


	return (locs)

prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
time.sleep(1)
i = 1

dir = "/home/pg/Desktop/CloudSaved/face_detection/dataset/masked"
memey = "/home/pg/Desktop/CloudSaved/face_detection/dataset/masked/new"
list = os.listdir(dir) # dir is your directory path
'''
picture = "{}/{:03}.jpg".format(dir, i)
picture = cv2.imread(picture)
print('{} {}'.format(picture, picture.shape[:2]))

'''
for i in range(len(list)):
	i = i +1
	picture = "{}/{:03}.PNG".format(dir, i)
	frame = cv2.imread(picture)
	(locs) = detect_and_predict_mask(frame, faceNet)

	valid = False
	for (box) in locs:
		try:
			if(box):
				valid = True
			else:
				valid = False
			#print("box: {} no:{}".format(box,i))
			
			#print('cor:{} i:{} name:{:03}.PNG array:{}'.format(box, i, i, valid))
			(startX, startY, endX, endY) = box
			img = frame[startY:endY, startX:endX]
			cv2.imwrite(picture, img)
			print("{:03}.PNG done".format(i))
			
		except:
			pass

print("All done")