#!/usr/bin/python
# -*- coding: UTF-8 -*-
from pydispatch import dispatcher
import my_gpio


SIGNAL_delete = 'addContactButton pressed!'
SIGNAL_exit = 'shutterButton pressed!'
delete_sender = object()
delete_flag = False

def setup_GPIO():
    """Setup GPIOs"""
    ADD_CONTACT_PIN = 20
    SHUTTER_PIN = 21
    GREEN_LED_PIN = 16
    RED_LED_PIN = 12
    gpio = my_gpio.MyGPIO([],ADD_CONTACT_PIN,SHUTTER_PIN,GREEN_LED_PIN,RED_LED_PIN)
    return gpio


def handle_delete(sender):

	delete_flag = True
	if delete_flag and 


if gpio.shutterButtonPressed():
	dispatcher.send(signal = SIGNAL_delete, sender = delete_sender)
	dispatcher.connect(handle_delete, signal = SIGNAL_delete, sender = dispatcher.Any)

