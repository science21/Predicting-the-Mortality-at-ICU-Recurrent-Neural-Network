import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from sklearn.preprocessing import MaxAbsScaler

def load_seizure_dataset(path, model_type):
	   
    df =pd.read_csv(path)
    data=df.drop(['y'],axis=1).values
    target=df['y'].values -1
    #scaler = MaxAbsScaler().fit(data)
    #data = scaler.transform(data)
    
    if model_type == 'MLP':
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')), torch.from_numpy(target.astype('float32')))
    elif model_type == 'CNN':
	    dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(1), torch.from_numpy(target.astype('long')))		
    elif model_type == 'RNN':
        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).unsqueeze(2), torch.from_numpy(target.astype('long')))
    else:
    	raise AssertionError("Wrong Model Type!")

    return dataset

"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.

"""

def calculate_num_features(seqs):	
    lst=set([a for x in seqs for y in x for a in y])
    maxnum=max(lst)
    return int(maxnum)+1

"""
	:param seqs:
	:return: the calculated number of features

	# TODO: Calculate the number of features (diagnoses codes in the train set)
"""

class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels, num_features):

        if len(seqs) != len(labels):
            raise ValueError("Seqs and Labels have different lengths")

        self.labels = labels
        seqlist=[]
        for i in seqs:
            k=len(i)
            arr=np.zeros((k,num_features))
            for j in list(range(k)):
                ind=i[j]
                ind1=[int(a) for a in ind]
                arr[j,ind1]=1
            seqlist.append(arr)

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
        self.seqs = seqlist  # replace this with your implementation.

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
        return self.seqs[index], self.labels[index]

"""
Args:
seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
labels (list): list of labels (int)
num_features (int): number of total features available
"""



def visit_collate_fn(batch):
    
    seqs = [a[0] for a in batch]
    labels = [a[1] for a in batch]
    length=[]
    for i in seqs:
        k=len(i)
        length.append(k)
    length.sort(reverse=True)
    seqs.sort(key=len, reverse=True)
    LEN=max(length)
    arr=np.zeros((LEN,901))
    seq_list=[arr]*len(seqs)
    for i in list(range(len(seqs))):
        shp = seqs[i].shape
        s1,s2 =shp
        seq_list[i][:s1,:s2]=seqs[i]
    seq_array = np.dstack(seq_list)

    seqs_tensor = torch.FloatTensor(seq_array)
    lengths_tensor = torch.LongTensor(length)
    
    
    labels_tensor = torch.LongTensor(labels)

    return (seqs_tensor, lengths_tensor), labels_tensor


"""
 DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
 Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

:returns
seqs (FloatTensor) - 3D of batch_size X max_length X num_features
lengths (LongTensor) - 1D of batch_size
 labels (LongTensor) - 1D of batch_size
 
 
# TODO: Return the following two things
# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
# TODO: 2. Tensor contains the label of each sequence
 
"""
