import numpy as np
import cv2
import picamera

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
    return frame

cam = setup_camera()

while True:
    img = capture_frame(cam)

    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(l)

    limg = cv2.merge((cl1,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imshow('hi',img)
    cv2.imshow('altered',final)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
