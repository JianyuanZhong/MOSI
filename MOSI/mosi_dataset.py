from torch.utils.data import DataLoader, Dataset
import torch

class MOSIDataset(Dataset):
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):

		X = torch.FloatTensor(self.X[idx])
		Y = torch.FloatTensor(self.Y[idx][0])

		return X, Y