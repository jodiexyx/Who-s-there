#!/usr/bin/python2 -B
from sys import argv
import sys
import numpy as np
import my_gpio
import traceback
import openface
from dlib import rectangle
import picamera
import cv2
import os
import subprocess
import new_tracker
from sklearn.svm import SVC
import sklearn.preprocessing
from wav_player import WavPlayer
import delete
import sys

CAMERA_WIDTH = 416
CAMERA_HEIGHT = 736
in_add_contact = False
in_delete = False

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
    ADD_CONTACT = 20
    SHUTTER = 21
    HAPTIC_L = 23
    HAPTIC_R = 24
    gpio = my_gpio.MyGPIO([HAPTIC_L, HAPTIC_R],ADD_CONTACT,SHUTTER)
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

def recognize(align,net,face,frame, classifier, le):
    rep = get_rep(align,net,face,frame)

    return classify(le, classifier, rep)

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
    camera.resolution = (CAMERA_HEIGHT, CAMERA_WIDTH)
    camera.vflip = False #might need to vertically flip in the future
    camera.rotation = 90
    camera.exposure_compensation = 10
    return camera

def capture_frame(camera):
    frame = np.empty((CAMERA_HEIGHT * CAMERA_WIDTH * 3,), dtype=np.uint8)
    camera.capture(frame, "bgr")
    frame = frame.reshape((CAMERA_WIDTH, CAMERA_HEIGHT, 3))
    frame = frame[:CAMERA_WIDTH, :CAMERA_HEIGHT, :]

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(l)

    limg = cv2.merge((cl1,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def setup_svm():
    labels_dict = None
    import pickle
    with open("/home/pi/MDP/label_database.pickle", 'rb') as handle:
        labels_dict = pickle.load(handle)    

    with open("/home/pi/MDP/encoding_base.pickle", 'rb') as base_encodings:
        encodings_dict = pickle.load(base_encodings)

    if len(labels_dict) < 50:
        labels_dict.update(encodings_dict)
    
    labels = []
    encodings = []

    #Create list with id per encoding
    for key, value in labels_dict.iteritems():
        for encoding in value:
            labels.append(key)
            encodings.append(encoding)

    le = sklearn.preprocessing.LabelEncoder().fit(labels)
    labelNumbers = le.transform(labels)

    classifier = SVC(C=100, kernel='linear', probability=True)
    classifier.fit(encodings, labelNumbers)

    return le, classifier, labels_dict

def classify(le, clf, rep):
    rep = rep.reshape(1, -1)

    prediction = clf.predict(rep)

    person = le.inverse_transform(prediction)
    print person[0]
    return person[0]

def verify_face(encodings, rep, name):
    for encoding_array in encodings[name]:
        
        min_dist = sys.maxsize
        dist = 1 - np.dot(rep, encoding_array)
        if dist < min_dist:
            min_dist = dist
    print min_dist
    print "Min Dist Qualifies:"
    print min_dist < 0.32
    return min_dist < 0.32

def vibrate_helper(gpio, face):
    if face[0]+0.5*face[2] < CAMERA_HEIGHT/3:
        #gpio.vibrate("left")
        print ("Left")
	mapping = [1]
	return mapping
    elif face[0]+0.5*face[2] < 2*CAMERA_HEIGHT/3:
        #gpio.vibrate("Center")
        print ("Center")
	mapping = [1,2]
	return mapping
    else:
        #gpio.vibrate("Right")
        print ("Right")
	mapping = [2]
	return mapping

def drive():
    global in_add_contact
    global in_delete
    camera = setup_camera()
    print ("Camera setup complete")
    gpio = setup_GPIO()
    print("GPIO setup complete")
    face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
    print("OpenCV Haar Caascade setup complete")
    tracker = new_tracker.Tracker(200)
    print ("Tracker setup complete")
    # TODO: Insert Coco/Jodie's unpickle of dict here
    labelEncoder, classifier, labels_dict = setup_svm()
    print ("SVM Classifier Setup")
    wavplayer = WavPlayer(0)
    print ("Wavplayer Setup Complete")
    align, net = setup_openface()
    print ("OpenFace setup complete")

    from save_new import addContact
    contact =  addContact(align, net, gpio, face_cascade, camera, wavplayer, capture_frame)

    from delete import deleteContact
    delete = deleteContact(gpio, wavplayer)

    wavplayer.play('/home/pi/MDP/audio/welcome.wav')

    while True:
        if gpio.addContactClicked():
            contact.main()
            labelEncoder, classifier, labels_dict = setup_svm()
            gpio.shutterClicked()
            gpio.addContactClicked()

        if gpio.shutterClicked():
            delete.main()
            labelEncoder, classifier, labels_dict = setup_svm()
            gpio.addContactClicked()
            gpio.shutterClicked()
        try:
            frame = capture_frame(camera)
            faces = detect_faces(face_cascade,frame)
            to_recognize = tracker.update(faces)
            #cv2.imshow('window', frame)
            #cv2.waitKey(1)
            #print to_recognize
            for face in to_recognize:
                print("someone new in frame, attempting to recognize")
                person = recognize(align,net,face,frame, classifier, labelEncoder)
                if person is not None:
                    if verify_face(labels_dict, get_rep(align,net,face, frame), person) and ('neg' not in person):
                        #wavplayer.play("contacts/" + person + ".wav")
                        mapping = vibrate_helper(gpio, face)
			wavplayer.play("/home/pi/MDP/contacts/" + person + ".wav", mapping_arg=mapping)

        except BaseException as e:
            print("Exiting program cleanly")
            gpio.cleanup()
            print(traceback.format_exc())
            break


if __name__ == '__main__':
    drive()
