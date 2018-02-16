from scipy.io import wavfile
from scipy.io.wavfile import read
import sounddevice as sd

class WavPlayer:
    def __init__(self,device):
        self.device = device

    def play(self,filename, mapping_arg=[1,2]):
        fs, data = wavfile.read(filename)
        sd.play(data, fs, device=self.device, mapping=mapping_arg)
        status = sd.wait()
        if status:
            #parser.exit('Error during playback: ' + str(status))
	    print("Error during playback")
#---HOW TO USE---
#from wav_player import WavPlayer
if __name__ == '__main__':
    w = WavPlayer(0)
    w.play("audio/no_person.wav")

