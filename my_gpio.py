"""
This classs assumes the haptic actuators are oriented like this:
* *    * *    * *
 *  or     or * * etc.
"""
import threading
import time
import RPi.GPIO as GPIO

class MyGPIO:
    def __init__(self,haptic_pins,add_contact_pin,shutter_pin):
        self.haptic_pins = haptic_pins
        #Use BCM numbering scheme
        GPIO.setmode(GPIO.BCM)
        #use internal pull DOWNS
        GPIO.setup(add_contact_pin, GPIO.IN,GPIO.PUD_UP)
        GPIO.setup(shutter_pin, GPIO.IN,GPIO.PUD_UP)

        GPIO.add_event_detect(add_contact_pin, GPIO.RISING,
                callback=self.addContactEdge, bouncetime=20)
        GPIO.add_event_detect(shutter_pin, GPIO.RISING,
                callback=self.shutterEdge, bouncetime=20)
        self.shutter_clicked = False
        self.add_contact_clicked = False
        for pin in haptic_pins:
            GPIO.setup(pin, GPIO.OUT)
            print("setting pin "+str(pin)+" as output")
            GPIO.output(pin,GPIO.HIGH)
        
    def cleanup(self):
        GPIO.cleanup()        

    def vibrate(self,direction):
        pins = []
        if direction == "left":
            pins.append(self.haptic_pins[0])
        elif direction == "right":
            pins.append(self.haptic_pins[1])
        else:
            pins.append(self.haptic_pins[0])
            pins.append(self.haptic_pins[1])
        for pin in pins:
            thread = threading.Thread(target=self.worker,args=(pin,))
            thread.start()

    def shutterClicked(self):
        if self.shutter_clicked:
            self.shutter_clicked = False
            return True
        else:
           return False

    def addContactClicked(self):
        if self.add_contact_clicked:
            self.add_contact_clicked = False
            return True
        else:
           return False

    def worker(self,pin):
        GPIO.output(pin,GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(pin,GPIO.HIGH)
        time.sleep(0.07)
        GPIO.output(pin,GPIO.LOW)
        time.sleep(0.05)
        GPIO.output(pin,GPIO.HIGH)
    def addContactEdge(self,channel):
        self.add_contact_clicked = True
    def shutterEdge(self,channel):
        self.shutter_clicked = True
