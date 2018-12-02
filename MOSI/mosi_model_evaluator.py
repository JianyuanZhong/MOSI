import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

import time
import numpy as np
# from attention_model import LSTM_custom,MOSI_attention_classifier
from mosi_model import MOSI_SENTIMENT_CLASSIFIER
# import gzip, cPickle
import matplotlib.pyplot as plt
# from mosi_helper import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# from mosi_dataset import MosiDataset
import mosi_data_util as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MosiEvaluator():
	
	def evaluate(self, dataloader, model):
		# data = MosiDataset(dataset['X'], dataset['Y'])
		# dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4)
		model.eval()
		predicted_y=[]
		target_y = []
		# correct = 0
		# total = 0
		with torch.no_grad():
			for i, data in enumerate(dataloader, 0):
				seq, label = data
				seq, label = seq.to(device), label.to(device)

				for idx, vec in enumerate(seq):
					x = util.get_unpad_data(vec)
					x = torch.FloatTensor(x).to(device)
					output = model(x)
					print(output[0])
					output = self.binaryClass(output[0])
					# label = self.binaryClass(label)
					
					predicted_y.append(output)
					target_y.append(label[idx])
					# predicted_y.extend(output)
					# target_y.extend(label)
				# if(output == label):
				# 	correct += 1
				# total += 1

		acc =  accuracy_score(target_y,predicted_y)
		f1 = f1_score(target_y,predicted_y)
		precision  = precision_score(target_y,predicted_y)
		recall = recall_score(target_y,predicted_y)

		return [acc,precision,recall,f1]

	def fiveClass(self, score):

			if(score[0][-1] > 2.0):
				return -2
                
			elif(score[0][-1] > 1.0) and (score[0][0] <= 2.0):
				return -1
                
			elif(score[0][-1] >= -1.0) and (score[0][0] <= 1.0):
				return 0
                
			elif(score[0][-1] >= -2.0) and (score[0][0] < 1.0):
				return 1
			else:
				return 2

	def binaryClass(self, score):
		# lst = []
		# for op in score:
		# 	if(op <= 0.5):
		# 		lst.append(0.0)
		# 	else:
		# 		lst.append(1.0)
		# return lst
		if (score <= 0.5):
			return 0.0
		else:
			return 1.0
