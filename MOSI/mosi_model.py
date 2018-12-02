import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
import mosi_data_util as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelIO():
	'''
	The ModelIO class implements a load() and a save() method that
	makes model loading and saving easier. Using these functions not
	only saves the state_dict but other important parameters as well from
	__dict__. If you instantiate from this class, please make sure all the
	required arguments of the __init__ method are actually saved in the class
	(i.e. self.<param> = param). 
	That way, it is possible to load a model with the default parameters and
	then change the parameters to correct values from stored in the disk.
	'''
	ignore_keys = ['_backward_hooks','_forward_pre_hooks','_backend',\
		'_forward_hooks']#,'_modules','_parameters','_buffers']
	def save(self, fout):
		'''
		Save the model parameters (both from __dict__ and state_dict())
		@param fout: It is a file like object for writing the model contents.
		'''
		model_content={}
		# Save internal parameters
		for akey in self.__dict__:
			if not akey in self.ignore_keys:
				model_content.update(self.__dict__)
		# Save state dictionary
		model_content['state_dict']=self.state_dict()
		torch.save(model_content,fout)

	def load(self,fin,map_location=None):
		'''
		Loads the parameters saved using the save method
		@param fin: It is a file-like obkect for reading the model contents.
		@param map_location: map_location parameter from
		https://pytorch.org/docs/stable/torch.html#torch.load
		Note: although map_location can move a model to cpu or gpu,
		it doesn't change the internal model flag refering gpu or cpu.
		'''
		data=torch.load(fin,map_location)
		self.__dict__.update({key:val for key,val in data.items() \
			if not key=='state_dict'})
		self.load_state_dict(data['state_dict'])

class MOSI_SENTIMENT_CLASSIFIER(nn.Module, ModelIO):
    
	def __init__(self, D_in, D_h, D_out, n_layers):
		super(MOSI_SENTIMENT_CLASSIFIER, self).__init__()
		self.D_in = D_in
		self.D_h = D_h
		self.D_out = D_out
		self.n_layers = n_layers
		self.lstm = nn.LSTM(D_in, D_h, num_layers=n_layers, dropout=0.1)
		self.fc1 = nn.Linear(D_h, D_out)

	def init_hidden(self, batch_size):
		hidden = (torch.zeros(self.n_layers, batch_size, self.D_h).to(device),
				  torch.zeros(self.n_layers, batch_size, self.D_h).to(device))
		return hidden
                
        
	def forward(self, inputs):
		batch_size = inputs.size(1)
		hidden = self.init_hidden(batch_size)
		output, hidden = self.lstm(inputs, hidden)
		label = F.relu(self.fc1(output))
		label = label.view(1, -1)
		return label

Language_index = range(0, 300)
Covarep_index = range(300, 374)
Facet_index = range(374, 421)

# Language_index = range(0, 3)
# Covarep_index = range(3, 6)
# Facet_index = range(6, 7)
class Mosi_Fusion(nn.Module, ModelIO):

	def __init__(self, D_language,  D_audio, D_video, D_H_language, D_H_audio, D_H_video, n_layers):
		super(Mosi_Fusion, self).__init__()
		self.n_layers = n_layers
		self.D_H_language = D_H_language
		self.D_H_audio = D_H_audio
		self.D_H_video = D_H_video
		self.D = D_H_language + D_H_audio + D_H_video
		self.lstm = nn.LSTM(D_language, D_H_language, num_layers=n_layers, dropout=0.1)
		self.f_video = nn.Linear(D_video, D_H_video)
		self.f_audio = nn.Linear(D_audio, D_H_audio)
		self.f_fusion = nn.Linear(self.D, 1)
		self.drop=nn.Dropout(0.1)

	def dropout(self):
		s_dict=self.f_video.state_dict()
		s_dict['weight']=self.drop(s_dict['weight'])
		self.f_video.load_state_dict(s_dict)

		s_dict=self.f_audio.state_dict()
		s_dict['weight']=self.drop(s_dict['weight'])
		self.f_audio.load_state_dict(s_dict)


	def forward(self, inputs):
		# inputs = util.get_unpad_data(inputs)
		# print(len(inputs))
		language = inputs[:,Language_index]
		language = language.unsqueeze(0)
		hidden = self.init_hidden(language.size(1))
		
		audio = inputs[:,Covarep_index]
		audio = reduce(torch.add, audio) / len(inputs)

		video = inputs[:,Facet_index]
		video = reduce(torch.add, video) / len(inputs)


		# print(language.shape)
		# print(hidden)
		# print(audio)
		# print(video)

		if self.training:
			self.dropout()

		x_language, hidden = self.lstm(language, hidden)
		# x_language = 
		x_audio = torch.sigmoid(self.f_audio(audio))
		x_video = torch.sigmoid(self.f_video(video))
		# x_concate = self.concatinate(x_language, x_audio, x_video)
		x_concate = torch.cat((x_language[-1, -1], x_audio, x_video))
		output = self.f_fusion(x_concate)
		return output

	def init_hidden(self, batch_size):
		hidden = (torch.zeros(self.n_layers, batch_size, self.D_H_language).to(device),
				  torch.zeros(self.n_layers, batch_size, self.D_H_language).to(device))
		return hidden

	def concatinate(self, x_language, x_audio, x_video):
		x_concate = torch.empty(x_language.size(0), x_language.size(1), self.D, dtype=torch.float)
		print(x_language[-1,-1])
		print(x_audio.size())
		print(x_video.size())
		for i in range(x_language.size(0)):
			for j in range(x_language.size(1)):
				concate = torch.cat((x_language[i][j], x_audio[i][j], x_video[i][j]))
				# print(concate.size())
				x_concate[i][j] = concate
		return x_concate.to(device)


if __name__ == '__main__':
	a = [[1,2,3,4,5,6,6], [2,3,4,5,3,5,4], [2,5,2,4,5,7,3], [7,8,3,9,2,4,1]]
	b = [[1,2,2,4,7,6,4], [2,6,4,3,3,5,3], [1,5,3,5,7,8,1], [7,2,1,9,1,4,3]]
	c = [[1,5,3,5,5,6,3], [3,3,4,6,3,5,5], [7,2,4,5,7,0,8], [7,0,3,0,2,4,0]]

	inputs = [a,b,c]
	inputs = Variable(torch.FloatTensor(inputs))
	label = [0,1,0]
	label = Variable(torch.FloatTensor(label))

	mosi_model = Mosi_Fusion(3,3,1,3,3,1,2)

	optimizer = optim.Adam(mosi_model.parameters(), lr=0.001)
	criterion = nn.BCEWithLogitsLoss()
	for epoch in range(100000):
		# output = mosi_model(inputs)
		# output = output[:,-1,-1]

		batch_losses = []
		for idx, vec in enumerate(inputs):
			x = util.get_unpad_data(vec)
			x = torch.FloatTensor(x).to(device)
			y = torch.FloatTensor([label[idx]]).to(device)
			y_hat = mosi_model(x)
			loss = criterion(y_hat, y)
			batch_losses.append(loss)
		batch_loss = reduce(torch.add, batch_losses) / len(batch_losses)
		batch_loss.backward()

		# loss = criterion(output, label)

		optimizer.step()
		optimizer.step()

		print(loss)
		

	

