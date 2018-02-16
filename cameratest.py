import picamera
import numpy as np

CAMERA_WIDTH = 416
CAMERA_HEIGHT = 736

camera = picamera.PiCamera()
camera.resolution = (CAMERA_HEIGHT, CAMERA_WIDTH)


for i in range(0, 50):
	frame = np.empty((CAMERA_HEIGHT * CAMERA_WIDTH * 3,), dtype=np.uint8)
	camera.capture(frame, "bgr")
	print "Captured!"
