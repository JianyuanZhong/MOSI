import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle as pkl 
from mosi_model_evaluator import MosiEvaluator
import datetime
import csv
from mosi_model import MOSI_SENTIMENT_CLASSIFIER, Mosi_Fusion
# from mosi_model import MOSI_SENTIMENT_CLASSIFIER
from mosi_dataset import MOSIDataset
import mosi_data_util as util



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_version="hidden_dim/"



def validation_loss(validationloader, mosi_model, criterion):
	mosi_model.eval()
	losses = []
	with torch.no_grad():
		for i, data in enumerate(validationloader, 0):
			seq, label = data
			seq, label = Variable(seq).to(device), Variable(label).to(device)
			batch_losses = []
			for idx, vec in enumerate(seq):
				x = util.get_unpad_data(vec)
				x = torch.FloatTensor(x).to(device)
				y = torch.FloatTensor([label[idx]]).to(device)
				y_hat = mosi_model(x)
				loss = criterion(y_hat, y)
				batch_losses.append(loss)
			batch_loss = reduce(torch.add, batch_losses) / len(batch_losses)
			losses.append(batch_loss.cpu().data.numpy())
			# output = mosi_model(seq)
			# output = output[:,-1,-1]
			# loss = criterion(output, label)
			# losses.append(loss.cpu().data.numpy())

	return losses



def train_epoch(trainloader, mosi_model, optimizer, criterion):
	# trainset, validationset, testset = loaddata()
	# trainset = MOSIDataset(trainset['X'], trainset['Y'])
	# trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
	# trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
	losses = []
	mosi_model.train()
	for i, data in enumerate(trainloader, 0):
		seq, label = data
		seq, label = Variable(seq).to(device), Variable(label).to(device)
		optimizer.zero_grad()
		batch_losses = []
		
		for idx, vec in enumerate(seq):
			x = util.get_unpad_data(vec)
			x = torch.FloatTensor(x).to(device)
			y = torch.FloatTensor([label[idx]]).to(device)
			y_hat = mosi_model(x)
			loss = criterion(y_hat, y)
			batch_losses.append(loss)
		batch_loss = reduce(torch.add, batch_losses) / len(batch_losses)
		batch_loss.backward()
		# print(label.type())
		# break
		# output = mosi_model(seq)
		# output = output[:,-1,-1]
		# loss = criterion(output, label)
	
		optimizer.step()
		losses.append(batch_loss.cpu().data.numpy())
	# print("epoch: Finished Loss: {}".format(losses))
	return losses

def plot_loss(epoch_trainning_losses, epoch_validation_losses, model_name):
	fig_name = "fig/"+model_name+".png"
	legend = ["train_loss", "validation_loss"]
	plt.plot(epoch_trainning_losses)
	plt.plot(epoch_validation_losses)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(legend, loc='upper right')
	title = "Losses plot for" + model_name
	plt.title(title)
	plt.savefig(fig_name)
	plt.close()

def evaluate_best_model(validationloader, testloader, model_name, params):
	
	evaluator = MosiEvaluator()
	model_file = model_version + "models/" + model_name + ".model"

	state_params =  {'D_language' : 300, 'D_audio' : 74, 'D_video' : 47, 'D_H_language' : 1, 'D_H_audio' : 1, 'D_H_video' : 1, 'n_layers' : 2}
	# d_face_param={'input_dim':1,'hidden_dim':1,'context_dim':1}

	best_model=Mosi_Fusion(**state_params)
	best_model.load(open(model_file,'rb'))
	best_model.to(device)
	# print(best_model)

	comment="validtion evaluation for best model: "+model_name
	print(comment)
	eval_valiation = evaluator.evaluate(validationloader, best_model)

	comment="test evaluation for best model: "+model_name
	print(comment)
	eval_test = evaluator.evaluate(testloader, best_model)

	# result = (eval_valiation + eval_test) / 2
	
	# save_results(model_name, [result], params)
	result = eval_valiation + eval_test
	save_results(model_name, result, params)
	return result


def save_results(model_name,eval_results,params):
	print (params)

	eval_results=[model_name]+eval_results

	for key,value in params.items():
		print (value)
		eval_results.append(value)

	result_csv_file = model_version+"results/all_results.csv"	
	with open(result_csv_file, 'a') as out_f:
		wr = csv.writer(out_f)
		wr.writerow(eval_results)
	out_f.close()


def train(trainset, validationset, testset, mosi_model, params):
	model_name = "params_" + str(params)
	modle_file = model_version + "models/" + model_name + ".model"
	mosi_model.to(device)

	optimizer = optim.Adam(mosi_model.parameters(), lr=params['learning_rate'])
	criterion = nn.BCEWithLogitsLoss()

	epoch_trainning_losses = []
	epoch_validation_losses = []
	num_epoch = 500

	best_validation_loss = np.inf 

	# mosi_model.to(device)

	for epoch in range(num_epoch):
		
		trainloader = util.get_data_loader(trainset[0], trainset[1])
		validationloader = util.get_data_loader(validationset[0], validationset[1])
		testloader = util.get_data_loader(testset[0], testset[1])

		train_loss = train_epoch(trainloader, mosi_model, optimizer, criterion)
		epoch_trainning_losses.append(np.mean(train_loss))

		validate_loss = validation_loss(validationloader, mosi_model, criterion)
		epoch_validation_losses.append(np.mean(validate_loss))

		print("ephoch: {} trainning loss: {} validation loss: {}".format(epoch, np.mean(train_loss), np.mean(validate_loss)))
		valid_mean = np.mean(validate_loss)
		if valid_mean < best_validation_loss:
			best_validation_loss = valid_mean
			mosi_model.save(open(modle_file, 'wb+'))

		if (epoch % 5 == 0):
			plot_loss(epoch_trainning_losses, epoch_validation_losses, model_name)

	trainloader = util.get_data_loader(trainset[0], trainset[1])
	validationloader = util.get_data_loader(validationset[0], validationset[1])
	testloader = util.get_data_loader(testset[0], testset[1])
	result = evaluate_best_model(validationloader, testloader, model_name, params)
	print("finished: {} Summary {} %".format(params, result))


