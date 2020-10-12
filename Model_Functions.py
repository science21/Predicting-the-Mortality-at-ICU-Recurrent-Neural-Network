import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset
from scipy import sparse
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_curve, auc, roc_auc_score

################  Model  ########################

class GRU(nn.Module):
	def __init__(self, dim_input):
		super(GRU, self).__init__()
		self.fc1 = nn.Linear(in_features=dim_input, out_features=32)
		self.rnn = nn.GRU(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
		self.fc2 = nn.Linear(in_features=64, out_features=2)
	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		max_length = seqs.size(1)
		seqs = torch.tanh(self.fc1(seqs))
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		seqs, h = self.rnn(seqs)
		seqs, lengths = pad_packed_sequence(seqs, batch_first=True)
		seqs = self.fc2(h[-1])
		return seqs

##################  Functions  #################################################

class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels
		seqs_list = []
		for i in range(len(seqs)):
			mtrx = sparse.lil_matrix((len(seqs[i]), num_features))
			for j in range(len(seqs[i])):
				rows = np.array([j])
				cols = np.array(seqs[i][j])
				mtrx[rows, cols] += np.ones((rows.size, cols.size))
			seqs_list.append(mtrx)
		self.seqs = seqs_list

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):

	batch.sort(key=lambda x: x[0].shape[0], reverse=True)
	seqs, labels = map(list, zip(*batch))
	max_rows = seqs[0].shape[0]
	max_columns = seqs[0].shape[1]
	lengths = []
	seqs_final = []
	for i in range(len(seqs)): 
		x = seqs[i].shape[0]
		lengths.append(x)
		if x < max_rows:
			zero_mtrx = np.zeros((max_rows-x,max_columns))
			seqs_mtrx = sparse.vstack((seqs[i], zero_mtrx)).toarray()
		else:
			seqs_mtrx = seqs[i].toarray()
		seqs_final.append(seqs_mtrx)
        
	seqs_tensor = torch.FloatTensor(seqs_final)
	lengths_tensor = torch.LongTensor(lengths)
	labels_tensor = torch.LongTensor(labels)

	return (seqs_tensor, lengths_tensor), labels_tensor
class AverageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
        
def compute_batch_accuracy(output, target):
	with torch.no_grad():

		batch_size = target.size(0)
		_, pred = output.max(1)
		correct = pred.eq(target).sum()

		return correct * 100.0 / batch_size
        
def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=100):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	model.train()

	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		data_time.update(time.time() - end)

		if isinstance(input, tuple):
			input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
		else:
			input = input.to(device)
		target = target.to(device)

		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()

		losses.update(loss.item(), target.size(0))
		accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

		if i % print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
				epoch, i, len(data_loader), batch_time=batch_time,
				data_time=data_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg


def evaluate(model, device, data_loader, criterion, print_freq=10):
	batch_time = AverageMeter()
	losses = AverageMeter()
	accuracy = AverageMeter()

	results = []

	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(data_loader):

			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			target = target.to(device)

			output = model(input)
			loss = criterion(output, target)

			batch_time.update(time.time() - end)
			end = time.time()

			losses.update(loss.item(), target.size(0))
			accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

			y_true = target.detach().to('cpu').numpy().tolist()
			y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
			results.extend(list(zip(y_true, y_pred)))

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
					i, len(data_loader), batch_time=batch_time, loss=losses, acc=accuracy))

	return losses.avg, accuracy.avg, results

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(train_losses)), train_losses, label='Training')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
    plt.title('Loss_Curve')
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Training')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
    plt.title('Accuracy_Curve')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    
    plt.show()
    pass
def plot_confusion_matrix(results, class_names):

    y_true, y_pred = zip(*results)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=class_names, yticklabels=class_names, title='Confusion Matrix', ylabel='True_Label', xlabel='Predicted_Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > (cm.max() / 2.) else "black")

    plt.show()
    pass

def predict_mortality(model, device, data_loader):
	model.eval()
	# TODO: Evaluate the data (from data_loader) using model,
	# TODO: return a List of probabilities
    
	probas = []
	with torch.no_grad():
		for i, (input, target) in enumerate(data_loader):
			if isinstance(input, tuple):
				input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
			else:
				input = input.to(device)
			out1 = model(input)
			out2 = nn.Softmax()(out1)
			out3 = out2.numpy()
			probas.append(out3[0][1])
	return probas

def plot_roc_and_auc_score(outputs, labels, title):
    false_positive_rate, true_positive_rate, threshold = roc_curve(labels, outputs)
    auc_score = roc_auc_score(labels, outputs)
    plt.plot(false_positive_rate, true_positive_rate, label = 'ROC curve, AREA = {:.4f}'.format(auc_score))
    plt.plot([0,1], [0,1], 'red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.legend(loc = 'lower right')
    plt.show()