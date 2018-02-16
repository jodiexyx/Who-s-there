#!/usr/bin/python3
import cv2
import os
import numpy as np

imgs_path = '/home/pi/MDP/vid_sim/'
images = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
images.sort(key=lambda x: (x[0],x[1]))

img_index = 0
def capture(output,format='rgb'):
	global img_index
	if(img_index == len(images)):
		img_index = 0
	#print('Reading image:',images[img_index])
	img = cv2.imread(imgs_path+images[img_index],cv2.IMREAD_COLOR)
	img_index += 1
	output[:] = img[:]

#testing using gui on raspberry pi
def unit_test():
	output = np.empty((240, 320, 3), dtype=np.uint8)
	while True:
		capture(output)
		cv2.imshow('herro',output)
		k = cv2.waitKey(0)
		if k==27:    # Esc key to stop
			break
	cv2.destroyAllWindows()

#unit_test()
