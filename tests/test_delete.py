#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import my_gpio
import time
from Queue import *
from delete import deleteContact
from wav_player import WavPlayer

def setup_GPIO():
    """Setup GPIOs"""
    ADD_CONTACT_PIN = 20
    SHUTTER_PIN = 21
    GREEN_LED_PIN = 16
    RED_LED_PIN = 12
    gpio = my_gpio.MyGPIO([],ADD_CONTACT_PIN,SHUTTER_PIN)
    return gpio

def load_pickle(path):
	import pickle
	with open(path, 'rb') as handle:
		b = pickle.load(handle)
	return b


def write_pickle(b, path):
	import pickle
	with open(path, 'wb') as handle:
		pickle.dump(b, handle, protocol = pickle.HIGHEST_PROTOCOL)


#path = "label_database.pickle"
#b = load_pickle(path)
#time.sleep(1)
#print b

player = WavPlayer(0)
gpio = setup_GPIO()
delete = deleteContact(gpio,player)
delete.main()

gpio.cleanup()
