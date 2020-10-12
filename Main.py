import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset
from scipy import sparse
import time
import matplotlib.pyplot as plt
from Model_Functions import GRU,VisitSequenceWithLabelDataset, visit_collate_fn, plot_learning_curves, train, evaluate, predict_mortality, plot_roc_and_auc_score

torch.manual_seed(0)
if torch.cuda.is_available():
	torch.cuda.manual_seed(0)

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/mortality/processed/mortality.seqs.train"
PATH_TRAIN_LABELS = "../data/mortality/processed/mortality.labels.train"
PATH_VALID_SEQS = "../../data/mortality/processed/mortality.seqs.validation"
PATH_VALID_LABELS = "../../data/mortality/processed/mortality.labels.validation"
PATH_TEST_SEQS = "../../data/mortality/processed/mortality.seqs.test"
PATH_TEST_LABELS = "../../data/mortality/processed/mortality.labels.test"
PATH_TEST_IDS = "../../../data/mortality/processed/mortality.ids.test"
PATH_OUTPUT = "../../output/mortality/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 50
BATCH_SIZE = 32
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

flat_list = lambda x: [item for list in x for item in list]
flat_list_distinct = set(flat_list(flat_list(train_seqs)))
num_features =  len(flat_list_distinct)

train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)

model = GRU(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))
test_prob = predict_mortality(best_model, device, test_loader)
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
plot_roc_and_auc_score(test_prob,test_labels, 'AUC score')