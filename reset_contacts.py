#!/usr/bin/python2
from sys import argv
import pickle
import subprocess

fresh_pickle = {}
with open('label_database.pickle', 'wb') as handle:
	pickle.dump(fresh_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

subprocess.call("rm -rf contacts/*.png contacts/*.wav",shell=True)
