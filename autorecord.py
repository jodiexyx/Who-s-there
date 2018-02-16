from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import argparse
from scipy.io import wavfile
from scipy.io.wavfile import read
import sounddevice as sd
import soundfile as sf
import numpy as np
from time import sleep


THRESHOLD = 7000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
info_path="/home/pi/MDP/info/"

##################previous code
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', type=int_or_str,
                    help='output device (numeric ID or substring)')
args = parser.parse_args()
if args.device is None:
    args.device = 0
#################################
def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    #print max(snd_data)
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the end of 'snd_data' of length 'seconds' (float)"
    #r = array('h', [0 for i in range(int(seconds*RATE))])   #python3 uses xrange;python2 uses range
    r = array('h',[0])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])#python3 uses xrange;python2 uses range
    return r

######################   
def playding():
    filename = '/home/pi/MDP/audio/ding.wav'
    fs, data = wavfile.read(filename)

    sd.play(data, fs, device=args.device)
    status = sd.wait()
    if status:
        parser.exit('Error during playback: ' + str(status))
########################


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
   
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=2
        )

    num_silent = 0
    snd_started = False

    r = array('h')

    Really_silent = 0
    

    while 1:
        # little endian, signed short

        snd_data = array('h', stream.read(CHUNK_SIZE,exception_on_overflow = False))
        if byteorder == 'big':
            snd_data.byteswap()
        silent = is_silent(snd_data)
        #print(silent)

        if silent:
            Really_silent +=1
         
        
        if silent and snd_started:
            num_silent += 1

        
        elif not silent and not snd_started:
            snd_started = True
        
    
        if snd_started and num_silent > 50:
            print('end recording!!')
            break

        if silent and Really_silent > 300:
            print('nothing detected')
            break
        else:
            r.extend(snd_data)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    if sample_width is None:
        return -1
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    return 0

def record_contact(id):
    playding()
    sleep(0.1)
    rc = record_to_file('/home/pi/MDP/contacts/' + str(id) + ".wav")
    sleep(0.1)
    playding()
    return rc

if __name__ == '__main__':
    print("please speak a word into the microphone")
    playding()
    sleep(0.1)
    id = 'hello'
    record_to_file(str(id) + ".wav")
    sleep(0.1)
    playding()
    print("done - result written to test.wav" + str(id))
