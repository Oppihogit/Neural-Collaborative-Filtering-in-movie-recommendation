import pandas as pd
import numpy as np
import torch.utils.data
import torch.nn as nn
import torch
import os
import datetime
import heapq
import warnings
import tqdm
from tabulate import tabulate

warnings.filterwarnings('ignore')
CUDA_LAUNCH_BLOCKING=1

#define cuda
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

#Load and process data
print("The system is loading.....")
OverallRank=np.load("Data/OverallRank.npy", allow_pickle=True)
title_r=['user_id','movie_id','rating','timestamp']
title_m=['movie_id','title','genres']
ratings = pd.read_table("dataset/ratings.dat", sep='::', header=None, names=title_r, engine='python')
movies= pd.read_table("dataset/movies.dat",sep='::',header=None,names=title_m,engine='python',encoding="ISO-8859-1")
user_negative_list = np.load('Data/user_negative_list.npy', allow_pickle=True).tolist()
Ranklist_Non_rs=OverallRank.tolist()

old_id = movies["movie_id"].tolist()
new_id = list(range(1, len(movies['movie_id']) + 1))
map_dic = dict(zip(old_id, new_id))

#reindex the movie_id of data, because there are some movies not in the movies file
ratings["movie_id"] = ratings["movie_id"].apply(lambda x: map_dic[x])
movies["movie_id"] = movies["movie_id"].apply(lambda x: map_dic[x])


num_users = max(ratings["user_id"])+1
num_items = max(ratings["movie_id"])+1


#define the layers of MLP network
num_factor = 8
layers = [num_factor*2, 64, 32, 16]

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
        MF_v = torch.mul(MF_user_e, MF_item_e)

        #Multi-layer perceptron model
        MLP_user_e = self.MLP_user_e(inputs[:, 0])
        MLP_item_e = self.MLP_item_e(inputs[:, 1])
        MLP_e = torch.cat([MLP_user_e, MLP_item_e], dim=-1)
        MLP_v=self.MLP_network(MLP_e)

        #Combine two vectors by a single layer network
        vector = torch.cat([MF_v, MLP_v], dim=-1)
        result = self.Combine_network(vector)
        return result


#load the pretrain model
model = NeuralCF(num_users, num_items, layers)
model.load_state_dict(torch.load('Pre_train/NCF_01.pkl'))
model.to(device)

def generate_recommendation(user_id,model):
    #create the feature according to the user_id
    index=user_id-1
    item_list=list(user_negative_list[index])
    user_list=[user_id]*(len(item_list))
    feature_list=list(zip(user_list,item_list))
    feature = list(map(lambda x: list(x), feature_list))
    a=torch.tensor(feature).to(device)

    #model output the prediction for input
    prediction=model(a)
    prediction=prediction.to("cpu")
    prediction_list=prediction.tolist()
    feature_prediction=list(zip(item_list,prediction_list))

    #sort the movies according to the value of prediction
    feature_prediction=sorted(feature_prediction,key=(lambda x:x[1]),reverse=True)

    #output the movie list for this user
    return [p[0] for p in feature_prediction]


def input_user_id():
    try:
        os.system('cls')
        print("        welcome,this is durflix")
        while True:
            user_id = int(input("        Enter your user_id (1-6040): "))
            if 0<user_id<=6040:
                return user_id
            else:
                print("        user_id must be the integer between 1 and 6040")

    except ValueError:
        print("        user_id must be the integer between 1 and 6040")
        return input_user_id()



def print_recommendation_non(prediction,p):
    os.system('cls')
    title=[]
    genre=[]

    #print 20 movies every pages
    print("\nMovies recommendation for you (Non-Personalised recommendations):")
    for i in prediction[p:p+20]:
        title.append(movies[movies["movie_id"] == i]['title'].values[0])
    for i in prediction[p:p+20]:
        genre.append(movies[movies["movie_id"] == i]['genres'].values[0])
    data={'movie id':prediction[p:p+20],
          'title':title,
          'genres':genre}
    print(tabulate(data,headers="keys"))
    page=p//20+1
    print(("\n                     ---------page %d---------")%(page))
    if p==0:
        print('\n      --------------------------   --Next page: enter N--     --exit: enter E--')
        print('\n      --------------- Enter the movie id to choose the movie ------------------\n')
    else:
        print('\n      --Previous page: enter P--   --Next page: enter N--     --exit: enter E--')
        print('\n      --------------- Enter the movie id to choose the movie ------------------\n')

    # check the input is vaild
    while True:
        commend = str(input("Enter your operation: "))
        try:
            commend=int(commend)
            if 0<commend<=6040 and commend in prediction[p:p+20]:
                return commend
            else:
                print("   Enter the correct movie_id please: ")
        except:
            if commend == "P" or commend == 'p':
                if p!=0:
                    return print_recommendation_non(prediction, p-20)
                else:
                    return print_recommendation_non(prediction, p)
            elif commend == "N" or commend == 'n':
                return print_recommendation_non(prediction, p+20)
            elif commend== "E" or commend == 'e':
                exit()
            else:
                print("Enter the correct command please: ")


def print_recommendation(prediction,p):
    os.system('cls')
    title=[]
    genre=[]
    print("\nUser_id : ", user_id)
    print("Movies recommendation for you (Personalised recommendations):")
    # print 20 movies every pages
    for i in prediction[p:p+20]:
        title.append(movies[movies["movie_id"] == i]['title'].values[0])
    for i in prediction[p:p+20]:
        genre.append(movies[movies["movie_id"] == i]['genres'].values[0])
    data={'movie id':prediction[p:p+20],
          'title':title,
          'genres':genre}
    print(tabulate(data,headers="keys"))
    page=p//20+1
    print(("\n                            ---------page %d---------")%(page))
    if p==0:
        print('\n      --------------------------   --Next page: enter N--     --exit: enter E--')
        print('\n      --------------- Enter the movie id to choose the movie ------------------\n')
    else:
        print('\n      --Previous page: enter P--   --Next page: enter N--     --exit: enter E--')
        print('\n      --------------- Enter the movie id to choose the movie ------------------\n')

    #check the input is vaild
    while True:
        commend = str(input("      Enter your operation: "))
        try:
            commend=int(commend)
            if 0<commend<=6040 and commend in prediction[p:p+20]:
                return commend
            else:
                print("Enter the correct movie_id please: ")
        except:
            if commend == "P" or commend=='p':
                if p!=0:
                    return print_recommendation(prediction, p-20)
                else:
                    return print_recommendation(prediction, p)
            elif commend == "N" or commend=='n':
                return print_recommendation(prediction, p+20)
            elif commend== "E" or commend=='e':
                exit()
            else:
                print("Enter the correct command please: ")

#User interface
os.system('cls')
print("                       Welcome to Durflix\n")
print("                   Please choose a mode to use")
print("        Enter 1 --- I have an user id (Personalised recommendations)")
print("        Enter 2 --- Use app as a visitor (Non-Personalised recommendations)")


#mode select
while True:
    mode=input("        Select your mode to use: ")
    if mode == '1' or mode == '2':
        break
    else:
        print("Enter the correct command please")

#Personalised recommendations model
if mode == '1':
    user_id=input_user_id()
    prediction=generate_recommendation(user_id,model)
    p=0
    item_s=print_recommendation(prediction,p)
    print(("Movie: %s is finished....")%(movies[movies["movie_id"] == item_s]['title'].values[0]))
    new_rating=input("Please rate this movie 1-5: ")
    print("Thanks for your rating, your feedback would help us improve the model")
    new_timestamp=str(datetime.datetime.now())
    new_data=[user_id,item_s,new_rating,new_timestamp]

    #the new data would be save to used in the next training
    np.save("Data/new_data.npy",new_data)
    print("new_data has been saved as Data/new_data.npy")

#Non-Personalised recommendations
else:
    item_s=print_recommendation_non(Ranklist_Non_rs,0)
    print(("Movie: %s is finished....")%(movies[movies["movie_id"] == item_s]['title'].values[0]))
    new_rating=input("Please rate this movie 1-5: ")
    print("Thanks for your rating, your feedback would help us improve the model")



