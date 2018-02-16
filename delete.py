#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import my_gpio
import time
from Queue import *
import select
from pydispatch import dispatcher

class deleteContact:
    def __init__(self,gpio,wavplayer):
        self.gpio = gpio
        self.wavplayer = wavplayer
        self.info_path = "/home/pi/MDP/contacts/"
        self.PICKLE_DIR = "/home/pi/MDP/label_database.pickle"
        self.all_contact = Queue(maxsize = 0)
        self.contact_info = self.load_pickle()
        self.delete_contact = []
        self.done = False
        self.cur_id = None
        self.SIGNAL_EXIT = "SHUTTER_CLICKED"
        self.SIGNAL_DELETE = "ADD_CONTACT_CLICKED"
        self.inDelete = False


    def load_pickle(self):
        import pickle
        print self.PICKLE_DIR
        with open(self.PICKLE_DIR, 'rb') as handle:
            b = pickle.load(handle)
        return b


    def write_pickle(self, b):
        import pickle
        with open(self.PICKLE_DIR, 'wb') as handle:
            pickle.dump(b, handle, protocol = pickle.HIGHEST_PROTOCOL)
    

    def delete(self):
        if len(self.delete_contact) == 0:
            return
        for tempid in self.delete_contact:
            self.contact_info.pop(tempid)
            os.remove(self.info_path + str(tempid) + ".wav")
            print("deleted contact:", tempid)
        self.write_pickle(self.contact_info)


    def get_id(self):
        f = self.load_pickle()
        for key in f:
            print key
            self.all_contact.put(key)


    def play_voice(self, id):
        self.wavplayer.play(self.info_path + str(id) + ".wav")


    def is_letter_input(self, letter):
        if select.select([sys.stdin,],[],[],0.0)[0]:
            input_char = sys.stdin.read(1)
            return input_char.lower() == letter.lower()
        return False

    def main(self):
        self.delete_contact = []
        print("Entering delete contact")
        self.wavplayer.play("/home/pi/MDP/audio/entering_delete.wav")
        self.done = False
        self.contact_info = self.load_pickle()

        time.sleep(1)
        self.get_id()
        print self.all_contact.qsize()
        if self.all_contact.empty():
            self.wavplayer.play("/home/pi/MDP/audio/exit_delete.wav")
            print("contact list is empty and exit delete")
            return
        while not self.all_contact.empty():
            if self.gpio.shutterClicked():
                break
            else:       
                self.cur_id = self.all_contact.get()
                print("current id is:", self.cur_id)
                self.play_voice(self.cur_id)
                time.sleep(0.75)
                if self.gpio.addContactClicked():
                    time.sleep(2)
                    print("want to delete:", self.cur_id)
                    self.wavplayer.play("/home/pi/MDP/audio/deleted.wav")
                    self.wavplayer.play("/home/pi/MDP/contacts/" + str(self.cur_id) + ".wav")
                    self.wavplayer.play("/home/pi/MDP/audio/continue_delete.wav")
                    self.delete_contact.append(self.cur_id)
        self.delete()
        self.wavplayer.play("/home/pi/MDP/audio/exit_delete.wav")
        print("exit delete")     



