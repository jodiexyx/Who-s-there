#!/usr/bin/python2
#this code is all just hacked together for wed demo
#TODO: make everything event driven and non-blocking
#TODO: make facial detection faster/use multiple cores
#TODO: pickle faces as face encodings (and images but only for ref)
from sys import argv
import numpy as np
np.set_printoptions(precision=2)
import my_gpio
import save
import traceback
# import config
import openface
from dlib import rectangle
import picamera
import cv2
import uuid
from PIL import Image
import os
import argparse
#import facial_tracking
#import tracking
import subprocess
from save_new import addContact
#from sklearn.svm import SVC
#import pandas as pd




def setup_camera():
    """Setup Camera"""
    #sim = True if len(argv) > 1 and argv[1] == '-s' else False
    sim = False
    camera = config.get_camera(sim)
    camera.resolution = (320, 240)
    camera.vflip = False #might need to vertically flip in the future
    camera.exposure_compensation = 10
    return camera


def setup_openface():
    """Setup OpenFace and return predictor and NN"""
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

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, args.imgDim)

    return align, net


def setup_GPIO():
    """Setup GPIOs"""
    ADD_CONTACT_PIN = 20
    SHUTTER_PIN = 21
    GREEN_LED_PIN = 16
    RED_LED_PIN = 12
    gpio = my_gpio.MyGPIO([],ADD_CONTACT_PIN,SHUTTER_PIN,GREEN_LED_PIN,RED_LED_PIN)
    return gpio


def getRep(face_img, frame, img_dim=96):
    """Get 128-vector representations"""
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
    x,y,w,h = transformBound(face_img, 0.85,0.95)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bb = rectangle(x,y,x+w,y+h)

    start = time.time()
    alignedFace = align.align(img_dim, frame, face_img,
        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    if alignedFace is None:
        print("Unable to align image")
        return None
   
    rep = net.forward(alignedFace)
    return rep

#The OpenCV face detection does not draw a perfect
#bounding rectangle around faces so this tries to correct that
def transformBound(rect,xs,ys):
    x,y,w,h = rect
    xt = int(round(x + (w-w*xs)/2))
    wt = int(round(w*xs))
    yt = int(round(y + (h-h*ys)/2))
    ht = int(round(h*ys))
    return (xt,yt,wt,ht)


def preprocess_images(raw_path, processed_path):
    # Util function provided by openface to align photos and preprocess
    subprocess.call(["""for N in {1..8}; do util/align-dlib.py %s align outerEyesAndNose %s --size 96 & done"""%(raw_path, processed_path)])


def create_feature_csv(work_dir, processed_path):
    # Util function to get feature vectors for IDs
    subprocess.call(['batch-represent/main.lua -outDir %s -data %s'%(work_dir, processed_path)])


def train(raw_path, processed_path, work_dir):
    """TODO: COME BACK TO THIS"""
    preprocess_images(raw_path, processed_path)
    create_feature_csv(work_dir, processed_path)
    fname = "{}/labels.csv".format(work_dir)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = np.loadtxt(open(fname, "rb"), delimiter=",", skiprows=1)[:,1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(work_dir)
    embeddings = np.loadtxt(open(fname, "rb"), delimiter=",", skiprows=1)[:,1]
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(work_dir)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

    out_dict = {}
    out_dict["labels"] = labels
    out_dict["encodings"] = embeddings
    return out_dict

"""
def recognize(le, clf, model_dir, face, frame):
    r = getRep(face, frame)

    rep = r[1].reshape(1, -1)
    bbx = r[0]

    start = time.time()
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]

    print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
    
    if confidence > .8:
        play_voice(person.decode('utf-8'))
        return True, person.decode('utf-8')
"""

def recognize(face, frame, label_encodings):
    r = getRep(face, frame)
    rep = r[1].reshape(1, -1)
    bbx = r[0]

    min_idx = 0
    min_dist = 1

    for i in range(0, len(label_encodings["labels"])):
        dist = 1 - np.dot(rep, label_encodings["encodings"][i])
        if dist < min_dist:
            min_dist = dist
            person = label_encodings["labels"][i]

    if min_dist < 0.25:
        play_voice(person.decode('utf-8'))
        return True, person.decode('utf-8')

def play_voice(person_name, entered_frame=True):
    print("{} came into frame".format(person_name))
    save.play_voice(person_name)


def load_model(work_dir):
    try:
        with open(work_dir, 'rb') as f:
            if sys.version_info[0] < 3:
                    (le, clf) = pickle.load(f)
            else:
                    (le, clf) = pickle.load(f, encoding='latin1')
        return le, clf
    except:
        return None, None


def add_contact(raw_path):
    id = save.generate_id()
    save.record_voice(id)
    frame = np.empty((240, 320, 3), dtype=np.uint8)
    if not os.path.exists(os.path.join(raw_path, str(id))):
        os.makedirs(os.path.join(raw_path, id))
    while True:
        if(gpio.addContactButtonPressed()):
            break
        if(gpio.shutterButtonPressed()):
            camera.capture(frame, format="rgb")
            frame = frame.reshape((240, 320, 3))
            frame = frame[:240, :320, :]
            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #OpenCV haarcascade face detection
            faces = faceCascade.detectMultiScale(
                grayImg,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            face = save.evaluate_faces(faces, frame, 240)
            if face is not None:
                im = Image.fromarray(frame[face[0]:face[0]+face[2], face[1], face[1]+face[3]])
                im.save(os.path.join(raw_path, id, str(uuid.uuid4())+".png"))


def drive():
    raw_path = "contact_photos/"
    processed_path = "processed_photos/"
    work_dir = "data/"
    faceCascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    camera = setup_camera()
    print ("Camera setup finished.")
    align, net = setup_openface()
    print ("OpenFace Setup")
    gpio = setup_GPIO()
    #le, clf = load_model(os.path.join(work_dir + "classifier.pkl"))
    tracker = facial_tracking.Tracker()
    print ("Tracker Initialized")

    frame = np.empty((240, 320, 3), dtype=np.uint8)

    args_dict = train(raw_path, processed_path, work_dir)

    recognize_args_dict = {}
    #recognize_args_dict['le'] = le
    #recognize_args_dict['clf'] = clf
    recognize_args_dict['model_dir'] = work_dir
    recognize_args_dict['label_encodings'] = args_dict

    while True:
        try:
            print("Capturing Photo")
            frame = np.empty((320 * 240 * 3,), dtype=np.uint8)
            camera.capture(frame, "rgb")
            frame = frame.reshape((240, 320, 3))
            frame = frame[:240, :320, :]

            grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #OpenCV haarcascade face detection
            faces = faceCascade.detectMultiScale(
                grayImg,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if(gpio.addContactButtonPressed()):
                addContact(align, net, gpio, faceCascade, camera)
                args_dict = train(raw_path, processed_path, work_dir)
                recognize_args_dict['label_encodings'] = args_dict
                #le, clf = load_model(os.path.join(work_dir + "classifier.pkl"))
                #recognize_args_dict['le'] = le
                #recognize_args_dict['clf'] = clf
            if le is not None:
                recognize_args_dict['frame'] = frame
                tracking._poll_tracker(tracker, faces, frame, recognize, recognize_args_dict)

        except BaseException as e:
            print("Exiting program cleanly")
            gpio.cleanup()
            print(traceback.format_exc())
            break

if __name__ == '__main__':
    drive()
