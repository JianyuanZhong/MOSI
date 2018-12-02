 # -*- coding: utf-8 -*-
import time
import torch
import gzip
import pickle as pkl 
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn

from mosi_model import MOSI_SENTIMENT_CLASSIFIER, Mosi_Fusion
from mosi_dataset import MOSIDataset
import mosi_train_helper as helper
import mosi_data_util as util
import numpy as np
# import cuda

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loaddata():
	with open('trainset.pikle', 'rb') as f:
		trainset = pkl.load(f)
	trainset = MOSIDataset(trainset['X'], trainset['Y'])
	
	with open('validationset.pikle', 'rb') as f:
		validationset = pkl.load(f)
	validationset = MOSIDataset(validationset['X'], validationset['Y'])
	
	with open('testset.pikle', 'rb') as f:
		testset = pkl.load(f)
	testset = MOSIDataset(testset['X'], testset['Y'])
	return trainset, validationset, testset

def load_COVAREP():
	trainf = gzip.open("/Users/zhongjianyuan/Desktop/research/Deep_Learning/normalized/COVAREP/train_matrix.pkl", "rb")
	train_data = pkl.load(trainf)

	testf = gzip.open("/Users/zhongjianyuan/Desktop/research/Deep_Learning/normalized/COVAREP/test_matrix.pkl", "rb")
	test_data = pkl.load(testf)

	validf = gzip.open("/Users/zhongjianyuan/Desktop/research/Deep_Learning/normalized/COVAREP/valid_matrix.pkl", "rb")
	valid_data = pkl.load(validf)

	print("Done! \nStart save data to files...")
	with open("trainset.pikle", "wb") as f:
		pkl.dump(train_data[0:15], f)
	with open("validationset.pikle", "wb") as f:
		pkl.dump(valid_data[0:15], f)
	with open("testset.pikle", "wb") as f:
		pkl.dump(test_data[0:15], f)
	print("Done!")

	trainf.close()
	testf.close()
	validf.close()

	return train_data, test_data, valid_data

def loadNewData():
	# with gzip.open('/home/jzhong9/MOSI/normalized/COVAREP/train_matrix.pkl', 'rb') as f:
	with gzip.open('/Users/zhongjianyuan/Desktop/research/Deep_Learning/fusion/data/trainset.pikle', 'rb') as f:
		trainset = pkl.load(f)
	# trainset = MOSIDataset(trainset['X'], trainset['Y'])
	
	# with gzip.open('/home/jzhong9/MOSI/normalized/COVAREP/valid_matrix.pkl', 'rb') as f:
	with gzip.open('/Users/zhongjianyuan/Desktop/research/Deep_Learning/fusion/data/validationset.pikle', 'rb') as f:
		validationset = pkl.load(f)
	# validationset = MOSIDataset(validationset['X'], validationset['Y'])
	
	# with gzip.open('/home/jzhong9/MOSI/normalized/COVAREP/test_matrix.pkl', 'rb') as f:
	with gzip.open('/Users/zhongjianyuan/Desktop/research/Deep_Learning/fusion/data/testset.pikle', 'rb') as f:
		testset = pkl.load(f)
	# testset = MOSIDataset(testset['X'], testset['Y'])
	return trainset, validationset, testset

# load_COVAREP()

# if __name__ == '__main__':
# 	D_IN = [300]
# 	D_HIDDEN = [128, 72, 84, 172, 28, 16, 32, 64]
# 	D_OUT = [1]
# 	Learning_Rate_List = [0.00066,0.0066,0.0033,0.0001,0.001,0.01]
# 	NUM_LAYERS = [2, 3, 4]

# 	print('loading dataset from files')
# 	trainset, validationset, testset = loaddata()
# 	print('Done!')
# 	# print(trainset['X'], trainset['Y'])
# 	# print(len(trainset['X']), len(trainset['Y']))


# 	# for a in trainset['X']:
# 	# 	a = torch.tensor(a)
# 	# 	print(a.size())

# 	print("Start training")
# 	for D_in in D_IN:
# 		for D_h in D_HIDDEN:
# 			for D_out in D_OUT:
# 				for num_layers in NUM_LAYERS:
# 					for lr in Learning_Rate_List:
# 						params = {'learning_rate' : lr, 'D_in' : D_in,'D_h' : D_h, 'D_out' : 1,'n_layers' : num_layers}
# 						print("training model: {}".format(params))
# 						mosi_model = MOSI_SENTIMENT_CLASSIFIER(D_in, D_h, D_out, num_layers)
# 						# print(mosi_model)
# 						# mosi_model.to(device)
# 						helper.train (trainset, validationset, testset, mosi_model, params)

if __name__ == '__main__':
	trainset, validationset, testset = loadNewData()
	D_language = 300
	D_audio = 74
	D_video = 47

	D_H_LANGUANGE = [128, 72, 56]
	D_H_AUDIO = [32, 72, 16]
	D_H_VIDEO = [32,  16]

	Learning_Rate_List = [0.00066, 0.0066, 0.01]

	for D_H_language in D_H_LANGUANGE:
		for D_H_audio in D_H_AUDIO:
			for D_H_video in D_H_VIDEO:
				for lr in Learning_Rate_List:
					params = {'learning_rate' : lr, 'D_H_language' : D_H_language, 'D_H_audio' : D_H_audio, 'D_H_video' : D_H_video, 'n_layers' : 2}
					mosi_model = Mosi_Fusion(D_language, D_audio, D_video, D_H_language, D_H_audio, D_H_video, 2)
					# mosi_model.to(device)
					helper.train (trainset, validationset, testset, mosi_model, params)

	
