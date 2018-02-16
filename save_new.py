#!/usr/bin/python
# -*- coding: UTF-8 -*-
import cv2
import subprocess
import uuid
import numpy as np
np.set_printoptions(precision=2)
import os
from sys import argv
import my_gpio
import traceback
import openface
from dlib import rectangle
import picamera
from PIL import Image
import argparse
import pygame
from autorecord import record_contact
import time
from pydispatch import dispatcher

class addContact:
    def __init__(self, align, net, gpio, faceCascade, camera, wavplayer, capture_frame):
        self.align = align
        self.net = net
        self.gpio = gpio
        self.faceCascade = faceCascade
        self.contact_list = []
        self.info_path = "/home/pi/MDP/contacts/"
        self.PICKLE_DIR = "/home/pi/MDP/label_database.pickle"
        self.camera = camera
        self.wavplayer = wavplayer
        self.done = False
        self.capture_frame_func = capture_frame

    def load_pickle(self):
        import pickle
        with open(self.PICKLE_DIR, 'rb') as handle:
            b = pickle.load(handle)
        return b


    def write_pickle(self, b):
        import pickle
        with open(self.PICKLE_DIR, 'wb') as handle:
            pickle.dump(b, handle, protocol = pickle.HIGHEST_PROTOCOL)


    def add_pickle(self, id, image_array):
        import pickle
        f = self.load_pickle()
        id = str(id)
        f[id] = image_array
        self.write_pickle(f)


    #arecord doesn't work on my mac but works on pi
    def record_voice(self, id):
        recorded_voice = False
        while not recorded_voice:
            print("recording...")
            self.wavplayer.play('/home/pi/MDP/audio/please_pronounce.wav')
            rc = record_contact(id)
            recorded_voice = (rc == 0)
        print("recorded!!!")


    #save image with id, one image each id
    def save_image(self, id, image_array):
        self.add_pickle(id, image_array) #add image_array into dict
        path = self.info_path + str(id) + ".png"
        cv2.imwrite(path, image_array) #save image_array into .png file
        print("encoded image saved!!!")


    #return the image_array corresponding to the given id
    def get_image(self, id):
        f = self.load_pickle()
        id = str(id)
        image_array = f[id]
        print("image=", image_array)
        return image_array


    #load and print the id-image pair for all contacts
    def get_contacts(self):
        contact_info = self.load_pickle()
        #print("contact_information==", contact_info)
        return contact_info


    def generate_id(self):
        contact_info = self.get_contacts()
        #contact_info={"id1", face_array1; "id2", face_array2;...;"idn", face_arrayn}
        while(True):
            id = uuid.uuid4()
            if(id not in contact_info):
                break
        return id

    def get_rep(self, face_bound, frame, img_dim=96):
        """Get 128-vector representations"""
        bb = None
        x,y,w,h = self.transform_bound(face_bound, 0.85,0.95)
        bb = rectangle(x,y,x+w,y+h)

        alignedFace = self.align.align(img_dim, frame, bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            print("Unable to align image")
            return None

        rep = self.net.forward(alignedFace)
        return rep


    #The OpenCV face detection does not draw a perfect
    #bounding rectangle around faces so this tries to correct that
    def transform_bound(self, rect,xs,ys):
        x,y,w,h = rect
        xt = int(round(x + (w-w*xs)/2))
        wt = int(round(w*xs))
        yt = int(round(y + (h-h*ys)/2))
        ht = int(round(h*ys))
        return (xt,yt,wt,ht)


    def detect_faces(self, face_cascade, frame):
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #OpenCV haarcascade face detection
        faces = face_cascade.detectMultiScale(
                grayImg,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
        return faces


    def capture_frame(self, camera):
        print "Enter Capture Frame"
        return self.capture_frame_func(camera)

    def main(self):
        print ("Entering Main")
        self.wavplayer.play("/home/pi/MDP/audio/entering_add.wav")

        contact_list = []
        id = self.generate_id()
        print ("Generated Id")


        while True:
            if self.gpio.shutterClicked():
                print("shutter clicked")
                frame = self.capture_frame(self.camera)
                print("Camera Captured")
                faces = self.detect_faces(self.faceCascade, frame)
                if faces is not None and len(faces) > 0:
                    print ("Appended")
                    self.wavplayer.play("/home/pi/MDP/audio/ding.wav")
                    encoding = self.get_rep(faces[0], frame)
                    contact_list.append(encoding)
                else:
                     # TODO: PLAY VOICE INSTEAD OF PRINT
                    print("No Faces Detected")
                    self.wavplayer.play("/home/pi/MDP/audio/no_person.wav")
            elif self.gpio.addContactClicked():
                break

        if(len(contact_list) == 0):
            self.wavplayer.play("/home/pi/MDP/audio/exit_add.wav")
            return
        self.add_pickle(id, contact_list)
        self.record_voice(id)
        self.wavplayer.play("/home/pi/MDP/audio/add_success.wav")
        print("contact added")
