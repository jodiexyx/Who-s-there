import pickle
import os
import csv
import numpy as np


with open('reps.csv', 'rb') as reps:
	reader = csv.reader(reps, delimiter=",")
	rep_list = list(reader)
	rep_array = numpy.array(rep_list).astype("float")

with open('labels.csv', 'rb') as labels:
	reader = csv.reader(labels, delimiter=",")
	labels_list = list(reader)
	labels_array = numpy.array(labels_list).astype("string")

print (rep_array)


