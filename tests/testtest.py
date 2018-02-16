#!/usr/bin/python2 -B
from sys import argv
import numpy as np
np.set_printoptions(precision=2)
import my_gpio
import traceback
import openface
from dlib import rectangle
import picamera
import cv2
import os
import subprocess
import new_tracker

def setup_openface():
    """Setup OpenFace and return predictor and NN"""
    fileDir = os.path.dirname(os.path.realpath(__file__))
    modelDir = '/home/pi/openface/models/'
    
    nn_image_dim = 96 
    dlib_face_predictor = os.path.join(modelDir, "dlib/shape_predictor_68_face_landmarks.dat")
    network_model = os.path.join(modelDir, 'openface/nn4.small2.v1.arm.t7')

    align = openface.AlignDlib(dlib_face_predictor)
    net = openface.TorchNeuralNet(network_model, nn_image_dim)

    return (align, net)

def setup_GPIO():
    """Setup GPIOs"""
    ADD_CONTACT_PIN = 20
    SHUTTER_PIN = 21
    GREEN_LED_PIN = 16
    RED_LED_PIN = 12
    gpio = my_gpio.MyGPIO([],ADD_CONTACT_PIN,SHUTTER_PIN,GREEN_LED_PIN,RED_LED_PIN)
    return gpio

def get_rep(align,net,face_bound, frame, img_dim=96):
    """Get 128-vector representations"""
    bb = None
    x,y,w,h = transform_bound(face_bound, 0.85,0.95)
    bb = rectangle(x,y,x+w,y+h)

    alignedFace = align.align(img_dim, frame, bb,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        print("Unable to align image")
        return None

    rep = net.forward(alignedFace)
    return rep

#The OpenCV face detection does not draw a perfect
#bounding rectangle around faces so this tries to correct that
def transform_bound(rect,xs,ys):
    x,y,w,h = rect
    xt = int(round(x + (w-w*xs)/2))
    wt = int(round(w*xs))
    yt = int(round(y + (h-h*ys)/2))
    ht = int(round(h*ys))
    return (xt,yt,wt,ht)

def recognize(align,net,face,frame,encodings):
    rep = get_rep(align,net,face,frame)

    #this subroutine finds the closest face using the dot product
    min_key = "naw"
    min_dist = 1
    for key,encoding in encodings.iteritems():
        dist = 1 - np.dot(rep, encoding)
        if dist < min_dist:
            min_dist = dist
            min_key = key
    if min_dist < 0.32:
        print("recognized ",min_key,", dist:",min_dist)
        save.play_voice(min_key)
        x_center = face[0]+face[1]/2
        if x_center <= 320/3.0:
            print("left")
        elif x_center <= 320/1.5:
            print("center")
        else:
            print("right")
        return min_key
    else:
        return None

def detect_faces(face_cascade,frame):
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

def add_contact(new_encoding,encodings):
    id = save.generate_id()
    save.save_contact(id,new_encoding)
    encodings[id] = new_encoding

def setup_camera():
    """Setup Camera"""
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    camera.vflip = False #might need to vertically flip in the future
    camera.exposure_compensation = 10
    return camera

def capture_frame(camera):
    frame = np.empty((320 * 240 * 3,), dtype=np.uint8)
    camera.capture(frame, "bgr")
    frame = frame.reshape((240, 320, 3))
    frame = frame[:240, :320, :]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(l)

    limg = cv2.merge((cl1,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def drive():
    camera = setup_camera()
    print ("Camera setup complete")
    gpio = setup_GPIO()
    print("GPIO setup complete")
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
    print("OpenCV Haar Caascade setup complete")
    tracker = new_tracker.Tracker(40)
    print ("Tracker setup complete")
    align, net = setup_openface()
    #align, net = None, None
    #print ("OpenFace setup complete")
    from save_new import addContact
    contact =  addContact(align, net, gpio, face_cascade, camera)    
    while True:
        try:
            #print("Capturing frame...")
            frame = capture_frame(camera)
            faces = detect_faces(face_cascade,frame)
            to_recognize = tracker.update(faces)
            
            if len(faces) == 0:
                if gpio.addContactButtonPressed():
                    #save.play_file("audio/no_person.wav",3)
                continue
            
            #print("detected "+str(len(faces))+" faces")
            #to_recognize = tracker.update(faces)

            for face in to_recognize:
                print("someone new in frame, attempting to recognize")
                #recognize(align,net,face,frame,encodings)

            if gpio.addContactButtonPressed():
                #rep = get_rep(align,net,faces[0],frame)
                contact.main()
            
        except BaseException as e:
            print("Exiting program cleanly")
            gpio.cleanup()
            print(traceback.format_exc())
            break

if __name__ == '__main__':
    drive()
