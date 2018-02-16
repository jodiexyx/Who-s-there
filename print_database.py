import pickle

with open('label_database.pickle', 'rb') as handle:
	b = pickle.load(handle)
	print b