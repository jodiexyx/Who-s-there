#!/usr/bin/python
# -*- coding: UTF-8 -*-
from sklearn import preprocessing, linear_model
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import random as rand
import math

PICKLE_DIR = '/Users/LiYushu/Documents/Umich_study /person_identification/PiCode/label_database.pickle'

def setup_svm():
    labels_dict = None
    import pickle
    with open("label_database.pickle", 'rb') as handle:
        labels_dict = pickle.load(handle)    

    with open("encoding_base.pickle", 'rb') as base_encodings:
        encodings_dict = pickle.load(base_encodings)


    labels_dict.update(encodings_dict)
    
    training_labels = []
    training_encodings = []

    test_labels = []
    test_encodings = []

	for label in labels_dict:
        rand_ints = []
        for i in range(0,3):
            index = rand.randint(0, len(labels_dict[label]))
            while index in rand_ints:
                print ("Looping")
                index = rand.randint(0, len(labels_dict[label]))
            rand_ints.append(index)

        for i in range(0, len(labels_dict[label])):
            if i in rand_ints:
                test_labels.append(label)
                test_encodings.append(labels_dict[label][i])
            else:
                training_labels.append(label)
                training_encodings.append(labels_dict[label][i])

	model = hard_C(training_encodings, training_labels)
    return classify(model, test_encodings, test_labels, labels_dict)


def hard_C(encodings, labels):
    svm = SVC(C = 100, kernel = 'linear', probability = True, random_state = 0)
    svm.fit(encodings, labels)
    return svm

def verify_face(encodings, rep, name):
    for encoding_array in encodings[name]:
        
        min_dist = sys.maxsize
        dist = 1 - np.dot(rep, encoding_array)
        if dist < min_dist:
            min_dist = dist
    print min_dist
    print "Min Dist Qualifies:"
    print min_dist < 0.32
    return min_dist < 0.32


def classify(model, te_d, te_l, labels_dict):
    num = len(te_d)
    pred = []
    for i in range(0, num):
        prediction = model.predict(te_d[i].reshape(1,-1)) 
        pred.append(prediction)

    count = 0.0
    error_count = 0.0
    for i in range(0, len(te_l)):
    	if verify_face(labels_dict, te_d[i], te_l[i]):
        	if te_l[i] == pred[i]:
            	count += 1.0
            else:
            	error_count += 1.0
    print("accuracy:", count / len(te_l))
    return count / len(te_l), error_count / len(te_l)

acc = []
error_list = []
for i in range(0, 100):
	acc, error_val = setup_svm()
    acc.append(acc)
    error_list.append(error_val)

print("average accuracy:", sum(acc) / len(acc))
print("false postives:", summ(error_list) / len(acc))

