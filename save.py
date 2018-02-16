import cv2
import subprocess
import uuid
import numpy as np
import os

#this is the path on disk
info_path="/home/pi/MDP/contacts"

#Pickle File for Users
PICKLE_DIR = 'label_database.pickle'


def load_pickle():
    import pickle
    with open(PICKLE_DIR, 'rb') as handle:
        b = pickle.load(handle)
    return b


def write_pickle(b):
    import pickle
    with open(PICKLE_DIR, 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)


def add_pickle(id, image_array):
    import pickle
    f=load_pickle()
    id=str(id)
    f[id]=image_array
    write_pickle(f)


def get_camera():
    # Camera to use for capturing images.
    # Use this code for capturing from the Pi camera:
    import picamera
    return picamera.OpenCVCapture()
    # Use this code for capturing from a webcam:
    #import webcam
    #return webcam.OpenCVCapture(device_id=0)


#arecord doesn't work on my mac but works on pi
def record_voice(id):
    print("recording...")
    subprocess.call(["arecord", "-d", "1", "-D", "plughw:1", "-r", "16000", info_path + str(id) + ".wav"])
    print("recorded!!!")


#aplay doesn't work on my mac but works on pi
def play_voice(id):
    #Popen used because it doesn't block
    subprocess.Popen(["aplay", info_path + str(id) + ".wav"])


#save image with id, one image each id
<<<<<<< HEAD
def save_image(id, image):
    add_pickle(id, image) #add image_array into dict
    path=info_path + str(id) + ".png" 
    cv2.imwrite(path, image) #save image_array into .png file
    print("image saved!!!")
=======
def save_image(id, image_array):
    add_pickle(id, image_array) #add image_array into dict
    path=info_path + str(id) + ".png"
    cv2.imwrite(path, image_array) #save image_array into .png file
    print("encoded image saved!!!")
>>>>>>> 48295cc30511d0bc76926bc6720a4f8428ae11bb


#return the image_array corresponding to the given id
def get_image(id):
    f=load_pickle()
    id=str(id)
    image_array=f[id]
    print("image=", image_array)
    return image_array


#load and print the id-image pair for all contacts
def get_contacts():
    contact_info=load_pickle()
<<<<<<< HEAD
    #print("contact_information==", contact_info) 
=======
    #print("contact_information==", contact_info)
>>>>>>> 48295cc30511d0bc76926bc6720a4f8428ae11bb
    return contact_info


def generate_id():
    contact_info=get_contacts()
    #contact_info={"id1", face_array1; "id2", face_array2;...;"idn", face_arrayn}
    while(True):
        id=uuid.uuid4()
        if(id not in contact_info):
            break
    #print("unique id=", id)
    return id


#for adding contact, save pictures as numpy array and voice(name) corresponding
#to the unique id
def save_contact(id, image_array):
    save_image(id, image_array)
    subprocess.call(["espeak","Please pronounce the contant's name"],stdout=open(os.devnull, 'w'),stderr=subprocess.STDOUT)
    record_voice(id)
