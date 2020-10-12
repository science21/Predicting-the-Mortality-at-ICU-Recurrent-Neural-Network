import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class MyMLP(nn.Module):
        def __init__(self):
                super(MyMLP, self).__init__()
                self.hidden1 = nn.Linear(178, 32)
                self.hidden2 = nn.Linear(32, 32)
                self.hidden3 = nn.Linear(32, 32)
                self.out = nn.Linear(32, 5)
                #self.hidden1 = nn.Linear(178, 16)
                #self.out = nn.Linear(16, 5)

        def forward(self, x):
                #x = torch.sigmoid(self.hidden1(x))
                #x = torch.sigmoid(self.hidden2(x))
                #x = torch.sigmoid(self.hidden3(x))
                x = F.relu(self.hidden1(x))
                x = F.relu(self.hidden2(x))
                x = F.relu(self.hidden3(x))
                x = self.out(x)
                return x


class MyCNN(nn.Module):
        def __init__(self):
                super(MyCNN, self).__init__()
                self.conv1 = nn.Conv1d(in_channels = 1, out_channels=6, kernel_size = 5)
                self.pool = nn.MaxPool1d(kernel_size=2)
                self.conv2 = nn.Conv1d(6, 16, 5)
                self.fc1 = nn.Linear(in_features = 16 * 41, out_features = 128)
                self.fc2 = nn.Linear(128, 5)
                #self.fc3 = nn.Linear(64, 5)


        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 41)
                x = F.relu(self.fc1(x))
                #x = F.relu(self.fc2(x))
                x = self.fc2(x)
                return x


class MyRNN(nn.Module):
        def __init__(self):
                super(MyRNN, self).__init__()
                self.rnn = nn.GRU(input_size = 1, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0)
                self.fc = nn.Linear(in_features = 16, out_features = 5)

        def forward(self, x):
                x, _ = self.rnn(x)
                x = self.fc(x[:,-1,:])
                return x


class MyVariableRNN(nn.Module):
        def __init__(self, dim_input):
                super(MyVariableRNN, self).__init__()
                # You may use the input argument 'dim_input', which is basically the number of features
                self.fc1 = nn.Linear(in_features = dim_input, out_features = 32)
                self.rnn = nn.GRU(input_size = 32, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0)
                #self.rnn = nn.LSTM(input_size = 32, hidden_size = 16, num_layers = 1, batch_first = True, dropout = 0)
                self.fc2 = nn.Linear(in_features = 16, out_features = 2)
                #self.fc1 = nn.Linear(in_features = dim_input, out_features = 64)
                #self.rnn = nn.LSTM(input_size = 64, hidden_size = 16, num_layers = 2, batch_first = True, dropout = 0)
                #self.fc2 = nn.Linear(in_features = 16, out_features = 2)

        def forward(self, input_tuple):
                # HINT: Following two methods might be useful
                # 'pack_padded_sequence' and 'pad_packed_sequence' from torch.nn.utils.rnn
                #print("input_tuple")
                #print(input_tuple)
                seqs, lengths = input_tuple
                #print("seqs")
                #print(seqs.size())
                #print("lengths")
                #print(lengths.size())
                #seqs.sort(key=lambda x:len(x) for x in seqs, reverse=True)
                #lengths.sort(reverse=True)
                max_length = max(lengths)
                #print("max_length")
                #print(max_length)
                #x = torch.tanh(self.fc1(seqs))
                x = F.relu(self.fc1(seqs))
                #print("after tanh")
                #print(x.size())
                x = pack_padded_sequence(x, lengths, batch_first=True)
                #print("post pack")
                #print(x.data.size())
                x,_ = self.rnn(x)
                #print("post gru")
                #print(type(x))
                x,_ = pad_packed_sequence(x, batch_first=True, total_length=max_length)
                #x,_ = pad_packed_sequence(x, batch_first=True)
                #print("post pad")
                #print(x.size())

                x = self.fc2(x[:,-1,:])
                #print("post fc")
                #print(x.size())
                



                return x
