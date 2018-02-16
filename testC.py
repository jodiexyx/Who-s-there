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

    #Create list with id per encoding
    #for key, value in labels_dict.iteritems():
        #for encoding in value:
            #labels.append(key)
            #encodings.append(encoding)

    for label in labels_dict:
        if "neg" in label:
            for i in range(0, len(labels_dict[label])):
                training_labels.append(label)
                training_encodings.append(labels_dict[label][i])
            continue

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


    #print training_labels

    #le = preprocessing.LabelEncoder().fit(training_labels)
    #traininglabelNumbers = le.transform(training_labels)
    # in real use, don't need to split training data, use them all as training
    #tr_d, tr_l, te_d, te_l = split_data(encodings, labelNumbers)
    #best_c = soft_C(training_encodings, traininglabelNumbers, test_encodings, test_labels)
    #model = get_model(training_encodings, traininglabelNumbers, best_c)
    model = hard_C(training_encodings, training_labels)
    return classify(model, test_encodings, test_labels)


def split_data(encodings, labels):
    num_all = len(labels)
    test_lab = labels[:3]
    test_data = encodings[:3]
    train_lab = labels[3:]
    train_data = encodings[3:]
    return train_data, train_lab, test_data, test_lab
    

def soft_C(tr_d, tr_l, te_d, te_l):
    acc_list = []
    c_list = []
    num_steps = 64
    for C in np.logspace(-15, 2, num_steps, base=2.0):
        svm = SVC(C = C, kernel = 'linear', probability = True, random_state = 0)
        score = cross_val_score(svm, tr_d, tr_l, cv = 5).mean()
        acc_list.append(score)
        c_list.append(C)
    print("c list:", c_list)
    print("accuracy list:", acc_list)
    maximum = max(acc_list)
    index = acc_list.index(maximum)
    print("the best training accuracy:", maximum)
    print("the best C is:", c_list[index])
    svm = SVC(C = c_list[index], kernel='linear', probability=True, random_state = 0)
    svm.fit(tr_d, tr_l)
    pred = svm.predict(te_d)
    score = accuracy_score(te_l, pred)
    print("soft margin score(the testing accuracy with best C):", score)
    return c_list[index]


def get_model(tr_d, tr_l, best_c):
    svm = SVC(C = best_c, kernel = 'linear', probability = True, random_state = 0)
    svm.fit(tr_d, tr_l)
    return svm
    

def hard_C(encodings, labels):
    svm = SVC(C = 100, kernel = 'linear', probability = True, random_state = 0)
    svm.fit(encodings, labels)
    return svm


def classify(model, te_d, te_l):
    num = len(te_d)
    pred = []
    for i in range(0, num):
        prediction = model.predict(te_d[i].reshape(1,-1)) 
        #predictions = model.predict_proba(te_d[i]).ravel()
        #maxI = np.argmax(predictions)

        #prediction = model.predict(te_d[i])
        #person = le.inverse_transform(prediction)
        #confidence = predictions[maxI]
        #print str(confidence)
        #print person
        #if confidence > .5:
            #.append(le[maxI])
        #else:
            #pred.append(-1)
        pred.append(prediction)
    count = 0.0
    for i in range(0, len(te_l)):
        if te_l[i] == pred[i]:
            count += 1.0
    print("accuracy:", count / len(te_l))
    return count / len(te_l)

acc = []
for i in range(0, 1000):
    acc.append(setup_svm())
print("average accuracy:", sum(acc) / len(acc))




