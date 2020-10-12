import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate, make_kaggle_submission
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyVariableRNN
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


torch.manual_seed(0)
if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/mortality/processed/mortality.seqs.train"
PATH_TRAIN_LABELS = "../data/mortality/processed/mortality.labels.train"
PATH_VALID_SEQS = "../data/mortality/processed/mortality.seqs.validation"
PATH_VALID_LABELS = "../data/mortality/processed/mortality.labels.validation"
PATH_TEST_SEQS = "../data/mortality/processed/mortality.seqs.test"
PATH_TEST_LABELS = "../data/mortality/processed/mortality.labels.test"
PATH_TEST_IDS = "../data/mortality/processed/mortality.ids.test"
PATH_OUTPUT = "../output/mortality/"

NUM_EPOCHS = 5
BATCH_SIZE = 32
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 4

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

num_features = calculate_num_features(train_seqs)
print("num features: " )
print(num_features)
train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)

model = MyVariableRNN(num_features)
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=0.003)
#optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.Adam(model.parameters())

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
model.to(device)
criterion.to(device)

#print("---------------------")

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
                #print("**********************8")
                best_val_acc = valid_accuracy
                torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))
#print(train_losses)
#print(valid_losses)
#print(train_accuracies)
#print(valid_accuracies)
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
plt.show()

best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))
# TODO: For your report, try to make plots similar to those in the previous task.
# TODO: You may use the validation set in case you cannot use the test set.


# TODO: Complete predict_mortality
def predict_mortality(model, device, data_loader):
        model.eval()
        # TODO: Evaluate the data (from data_loader) using model,
        # TODO: return a List of probabilities
        #test_loss, test_accuracy, test_results = evaluate(model, device, data_loader, criterion)
        #print("test_results")
        #print(test_results)
        #probas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        probas = []
        with torch.no_grad():
                for i, (input, target) in enumerate(data_loader):

                        if isinstance(input, tuple):
                                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
                        else:
                                input = input.to(device)

                        output = model(input)
                        #print(output)
                        output = nn.Softmax()(output)
                        #print(output)
                        output = output.numpy().tolist()
                        #print(output)
                        #print(output[0][1])
                        #print(output[:,1])
                        probas.append(output[0][1])
        
        return probas

test_prob = predict_mortality(best_model, device, test_loader)
#test_prob = predict_mortality(best_model, device, valid_loader)
test_id = pickle.load(open(PATH_TEST_IDS, "rb"))
test_label = pickle.load(open(PATH_TEST_LABELS, "rb"))
fpr, tpr, _ = roc_curve(test_label, test_prob)
roc_auc = auc(fpr, tpr)
print("roc_auc")
print(roc_auc)  #0.67346
make_kaggle_submission(test_id, test_prob, PATH_OUTPUT)

test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion)
#print("test_results")
class_names = ['alive','dead']
plot_confusion_matrix(test_results, class_names)
#plt.show()

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()