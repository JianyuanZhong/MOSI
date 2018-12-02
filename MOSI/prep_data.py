from mmsdk import mmdatasdk
import numpy as np
import gzip
import pickle as pkl 

def myavg(intervals,features):
        return np.average(features,axis=0)

def load_mosi():
	highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
	highlevel.align('glove_vectors',collapse_functions=[myavg])
	highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels,'cmumosi/')
	highlevel.align('Opinion Segment Labels')
	return highlevel

def prep_text_data_five_class(highlevel):
	glove_vectors = []
	labels = []
	for key, value in highlevel.computational_sequences["glove_vectors"].data.items():
		print("extracting vector: {}".format(key))
		glove_vectors.append(value['features'])

	for key, value in highlevel.computational_sequences["Opinion Segment Labels"].data.items():
		print("extracting label: {}".format(key))
		labels.append(value['features'])
	return glove_vectors, labels


def prep_text_data_binary_class(highlevel):
	glove_vectors = []
	labels = []
	for key, value in highlevel.computational_sequences["glove_vectors"].data.items():
		print("extracting vector: {}".format(key))
		glove_vectors.append(value['features'])

	for key, value in highlevel.computational_sequences["Opinion Segment Labels"].data.items():
		print("extracting label: {}".format(key))
		if(value['features'][0][0] < 0):
			labels.append([[0.0]])
		else:
			labels.append([[1.0]])
	return glove_vectors, labels

def prep_data(glove_vectors, labels):
	trainset = {'X' : glove_vectors[:1098], 'Y' : labels[:1098]}
	validationset = {'X' : glove_vectors[1099:1468], 'Y' : labels[1099:1468]}
	testset = {'X' : glove_vectors[1469:2198], 'Y' : labels[1469:2198]}
	return trainset, validationset, testset

def prep_dummy_data():
	print('Loading data from mmdatasdk...')
	highlevel = load_mosi()
	glove_vectors, labels = prep_text_data_binary_class(highlevel)
	trainset = {'X' : glove_vectors[:20], 'Y' : labels[:20]}
	validationset = {'X' : glove_vectors[21:31], 'Y' : labels[21:31]}
	testset = {'X' : glove_vectors[31:41], 'Y' : labels[31:41]}
	print("Done! \nStart save data to files...")
	with open("trainset.pikle", "wb") as f:
		pkl.dump(trainset, f)
	with open("validationset.pikle", "wb") as f:
		pkl.dump(validationset, f)
	with open("testset.pikle", "wb") as f:
		pkl.dump(testset, f)
	print("Done!")
	return trainset, validationset, testset

def save_data_five_class():
	print('Loading data from mmdatasdk...')
	highlevel = load_mosi()
	glove_vectors, labels = prep_text_data_five_class(highlevel)
	trainset, validationset, testset= prep_data(glove_vectors, labels)
	print("Done! \nStart save data to files...")
	with open("trainset.pikle", "wb") as f:
		pkl.dump(trainset, f)
	with open("validationset.pikle", "wb") as f:
		pkl.dump(validationset, f)
	with open("testset.pikle", "wb") as f:
		pkl.dump(testset, f)
	print("Done!")

def save_data_binary_class():
	print('Loading data from mmdatasdk...')
	highlevel = load_mosi()
	glove_vectors, labels = prep_text_data_binary_class(highlevel)
	trainset, validationset, testset= prep_data(glove_vectors, labels)
	print("Done! \nStart save data to files...")
	with open("trainset.pikle", "wb") as f:
		pkl.dump(trainset, f)
	with open("validationset.pikle", "wb") as f:
		pkl.dump(validationset, f)
	with open("testset.pikle", "wb") as f:
		pkl.dump(testset, f)
	print("Done!")

if __name__=='__main__':
	save_data_binary_class()


