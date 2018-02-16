#!/usr/bin/python2 -B
import cv2
import traceback
import picamera
import numpy as np

def transform_bound(rect,xs,ys):
    x,y,w,h = rect
    xt = int(round(x + (w-w*xs)/2))
    wt = int(round(w*xs))
    yt = int(round(y + (h-h*ys)/2))
    ht = int(round(h*ys))
    return (xt,yt,wt,ht)

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

def setup_camera():
    """Setup Camera"""
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    camera.vflip = False #might need to vertically flip in the future
    #camera.exposure_compensation = 10
    return camera

def capture_frame(camera):
    frame = np.empty((320 * 240 * 3,), dtype=np.uint8)
    camera.capture(frame, "rgb")
    frame = frame.reshape((240, 320, 3))
    frame = frame[:240, :320, :]
    return frame

camera = setup_camera()
face_cascade = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
found = False
tracker = None
while True:
        try:
            print("Capturing frame...")
            frame = capture_frame(camera)

            if found:
                ok,bbox = tracker.update(frame)
                print("tracker found bb of:",ok,bbox)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else:
                faces = detect_faces(face_cascade,frame)
                if len(faces) == 0:
                    continue
                tracker = cv2.Tracker_create("MIL")
                print("initiing tracker:",faces[0])
                tracker.init(frame, tuple(faces[0]))
                found = True
            cv2.imshow('window',frame)
            cv2.waitKey(1)
        
        except BaseException as e:
            print("Exiting program cleanly")
            print(traceback.format_exc())
            break

