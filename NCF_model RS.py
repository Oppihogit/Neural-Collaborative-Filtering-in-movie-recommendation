import pandas as pd
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch
from tqdm import tqdm
import heapq
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
import warnings

warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING=1

#define cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#parameter
topK = 10
num_factor = 8
num_negatives = 4
batch_size = 64
lr = 0.001

#load processed data_set
title_r=['user_id','movie_id','rating','timestamp']
train = np.load('Data/train_set.npy', allow_pickle=True).tolist()
test_set=np.load('Data/processed_testset.npy', allow_pickle=True)
user_negative_list = np.load('Data/user_negative_list.npy', allow_pickle=True).tolist()
ratings = pd.read_table("dataset/ratings.dat", sep='::', header=None, names=title_r, engine='python')


num_sample=len(train)
num_negative=4
label=[1]*num_sample


#append the negative instance to the trainset
for i in tqdm(range(num_sample)):
    u=train[i][0]
    it=random.sample(user_negative_list[u-1],num_negative)
    for j in it:
        train.append([u,j])
        label.append(0)

train_x=np.array(train)
label=np.array(label)
train_dataset = TensorDataset(torch.tensor(train_x), torch.tensor(label).float())
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_users = max(ratings["user_id"])+1
num_items = max(train_x[:,1])+1

#define the layers of MLP network
layers = [num_factor*2, 64, 32, 16]

#NeuralCF model
class NeuralCF(nn.Module):
    def __init__(self, num_user, num_item, layers):
        super(NeuralCF, self).__init__()
        #embedding dimension
        e_dim=layers[0] // 2

        #Embedding for Matrix factorisation model
        self.MF_user_e = nn.Embedding(num_embeddings=num_user, embedding_dim=e_dim)
        self.MF_item_e = nn.Embedding(num_embeddings=num_item, embedding_dim=e_dim)

        # Embedding for Multi-layer perceptron model
        self.MLP_user_e = nn.Embedding(num_embeddings=num_user, embedding_dim=e_dim)
        self.MLP_item_e = nn.Embedding(num_embeddings=num_item, embedding_dim=e_dim)

        #Multi-layer perceptron Network
        self.MLP_network = nn.Sequential(nn.Linear(layers[0], layers[1]),
                                         nn.ReLU(),
                                         nn.Linear(layers[1], layers[2]),
                                         nn.ReLU(),
                                         nn.Linear(layers[2], layers[3]),
                                         nn.ReLU(),
                                         nn.Linear(layers[3], e_dim))

        #Single layer Network to combine two result of models
        self.Combine_network = nn.Sequential(nn.Linear(layers[0], 1),
                                             nn.Sigmoid())


    def forward(self, inputs):
        #inputs is a batch of data: inputs[0,1]
        #trans inputs as long type
        inputs = inputs.long()

        #Matrix factorisation model
        MF_user_e = self.MF_user_e(inputs[:, 0])
        MF_item_e = self.MF_item_e(inputs[:, 1])
        MLP_user_e = self.MLP_user_e(inputs[:, 0])
        MLP_item_e = self.MLP_item_e(inputs[:, 1])

        MF_v = torch.mul(self.MF_user_e(inputs[:, 0]), self.MF_item_e(inputs[:, 1]))
        MLP_e = torch.cat([MLP_user_e, MLP_item_e], dim=-1)
        MLP_v=self.MLP_network(MLP_e)

        #Combine two vectors by a single layer network
        vector = torch.cat([MF_v, MLP_v], dim=-1)
        result = self.Combine_network(vector)
        return result


#evaluate(Hit_rate, MAE, NDCG)
def evaluate_model(model, test_set, K):
    def hit_check(true_instance, ranklist):
        if true_instance in ranklist:
            return 1
        return 0

    def MAE(true_instance, prediction_list_sorted):
        error=abs(float(true_instance)-1)
        return error

    def NDCG(true_instance, prediction_list_sorted): #Normalize Discounted Cumulative Gain
        index = prediction_list_sorted.index(true_instance)
        return np.log(2) / np.log(index + 2)

    hit_times = []
    MAE_list=[]
    NDCG_list=[]
    for i in range(len(test_set)):
        feature_list = list(map(lambda x: list(x), test_set[i][0]))
        feature = torch.tensor(feature_list).to(device)

        #get the prediction for every user
        prediction=model(feature)
        prediction=prediction.cpu()
        prediction_list = [p[0] for p in prediction]
        true_instance = prediction_list[0]
        ranklist = heapq.nlargest(K, prediction_list)
        hit_times.append(hit_check(true_instance, ranklist))
        prediction_list_sorted=sorted(prediction_list,reverse=True)
        MAE_list.append(MAE(true_instance,prediction_list_sorted))
        NDCG_list.append(NDCG(true_instance, prediction_list_sorted))


    return np.mean(hit_times),np.mean(MAE_list),np.mean(NDCG_list)

#define the model
model = NeuralCF(num_users, num_items, layers)
model.to(device)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


epochs = 20
t=0
topk=10
best_hit_rate=0
output_list = []

#Train NCF model
while t<epochs:
    model.train()
    loss_total = 0.0
    #
    for s, (feature, label) in enumerate(tqdm(trainloader), 1):
        feature= feature.to(device)
        label=  label.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(feature), label.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        if s % 5000 == 0:
            aver_loss= loss_total / s
            print(("Step = %d || loss: %.3f") %(s, aver_loss))
    t += 1

    #Evaluation the model after one epoch
    model.eval()
    hit_rate,MAE,ndcg = evaluate_model(model, test_set, topK)

    #Save the parameter if model got a better score
    if hit_rate > best_hit_rate:
        best_epoch, best_hit_rate, best_accuracy = t, hit_rate, MAE
        torch.save(model.state_dict(), 'Pre_train/NCF_02.pkl')

    print(("\nEPOCH = %d, loss = %.3f, MAE = %.3f, hr = %.3f, ndcg = %3f") % (t, loss_total / s, MAE, hit_rate, ndcg))
    output_list.append([t, loss_total / s, MAE, hit_rate, ndcg])

#save the recording of the training process
np.save('Data/train_output', output_list)
print('Finished Training...')