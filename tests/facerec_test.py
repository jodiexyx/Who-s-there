#!/usr/bin/env python2

import time
import argparse
import cv2
import itertools
import os
import numpy as np
np.set_printoptions(precision=2)
import openface
from dlib import rectangle
import picamera

#this is used to see which operations are taking a long time
start = time.time()

#set up pi camera
camera = picamera.PiCamera()
camera.resolution = (320, 240)
camera.vflip = False #might need to vertically flip in the future
#camera.exposure_compensation = 10

#set up opencv face detection
faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

#set up openface and dlib
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = '/home/pi/openface/models/'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.arm.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
args = parser.parse_args()

print("Initialization took {} seconds.".format(
    time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
print("Loading the dlib and OpenFace models took {} seconds.".format(
    time.time() - start))

#The OpenCV face detection does not draw a perfect
#bounding rectangle around faces so this tries to correct that
def transformBound(rect,xs,ys):
    x,y,w,h = rect
    xt = int(round(x + (w-w*xs)/2))
    wt = int(round(w*xs))
    yt = int(round(y + (h-h*ys)/2))
    ht = int(round(h*ys))
    return (xt,yt,wt,ht)

#this function returns the face represented as a 128-vector
def getRep(rgbImg):
    #print("Processing IMG")

    start = time.time() 
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #OpenCV haarcascade face detection
    faces = faceCascade.detectMultiScale(
        grayImg,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    bb = None
    if len(faces) == 0:
        #print("Haar detected no faces")
        return None
    else:
        print("Found face, processing")
        x,y,w,h = transformBound(faces[0],0.85,0.95)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        bb = rectangle(x,y,x+w,y+h)
    #This was the old facial detection algorithm
    #bb = align.getLargestFaceBoundingBox(rgbImg)
    #print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        print("Unable to align image")
        return None
    #print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    #print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
    return rep

frame = np.empty((240, 320, 3), dtype=np.uint8)
encodings = []

idx = 0
while True:
    # Capture frame-by-frame using py2's weird array conversions 
    frame = np.empty((320 * 240 * 3,), dtype=np.uint8)
    camera.capture(frame, 'rgb')
    frame = frame.reshape((240, 320, 3))
    frame = frame[:240, :320, :]
    #get the faces encoding as a 128-vector
    rep = getRep(frame)
    if rep is not None:
        if idx < 5:
             idx += 1
             encodings.append(rep)
        else:
            #this subroutine finds the closest face using the dot product
            min_idx = 0
            min_dist = 1
            for i,encoding in enumerate(encodings):
                dist = 1 - np.dot(rep, encoding)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            if min_dist < 0.25:
                print("image most similar to: ",min_idx,", dist:",min_dist)
    #show the face with face-rectangle on the gui
    #cv2.imshow('Detection', frame)
    if idx < 5:
        raw_input("Press Enter to continue...\n") 
