import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def process_rating():

    #split the dataset according to the timestamp
    def get_train(x):
        df = x.sort_values(by='timestamp',ascending=True)
        df.drop(df.tail(3).index,inplace=True)
        return df

    def get_test(x):
        df = x.sort_values(by='timestamp',ascending=True)
        return df.tail(3)

    path = 'dataset/'
    title_r = ['user_id', 'movie_id', 'rating', 'timestamp']
    title_m = ['movie_id', 'title', 'genres']
    ratings = pd.read_table(path + "ratings.dat", sep='::', header=None, names=title_r, engine='python')
    movies = pd.read_table(path + "movies.dat", sep='::', header=None, names=title_m, engine='python',
                           encoding="ISO-8859-1")

    #because there are some movies_id missing in the movie file, reindex all movie_id
    old_id=movies["movie_id"].tolist()
    new_id=list(range(1,len(movies['movie_id'])+1))
    map_dic=dict(zip(old_id,new_id))
    ratings["movie_id"]=ratings["movie_id"].apply(lambda x:map_dic[x])


    #user_negative_list which stored the movie which the user have not viewed before
    num_user = max(ratings["user_id"])
    test_set=ratings.groupby(by="user_id").apply(get_test)
    train_set=ratings.groupby(by="user_id").apply(get_train)
    user_positive_list = []
    for i in range(num_user):
        user_positive_list.append(train_set[train_set['user_id']==i+1]["movie_id"].tolist())
    movie_list= new_id
    print(len(movie_list),max(movie_list))
    user_negative_list=[]
    for i in user_positive_list:
        user_negative_list.append((set(movie_list)-set(i)))
    np.save("Data/user_negative_list.npy", user_negative_list)


    # user_negative_list which stored the movie which the user have not viewed before
    user=train_set["user_id"].tolist()
    item=train_set["movie_id"].tolist()
    train_set=list(zip(user,item))
    np.save("Data/train_set.npy", train_set)


    # create test_set
    user=test_set["user_id"].tolist()
    item=test_set["movie_id"].tolist()
    test_set=list(zip(user,item))
    np.save("Data/test_set.npy", test_set)


def create_testset():
    #load the test data
    test_set=np.load("Data/test_set.npy",allow_pickle=True)
    user_negative_list = np.load('Data/user_negative_list.npy', allow_pickle=True).tolist()
    i=0
    process_test_list=[]

    #append the negative instance(movies which user never viewed) to the test_set
    while i < len(test_set):
        a=test_set[i:i+3]
        u=a[0][0]
        true_item=a[:,1]
        negative_item=random.sample(user_negative_list[u-1],97)
        item_list=list(true_item)+negative_item
        negative_user=[u]*100
        feature=list(zip(negative_user,item_list))
        label=[1]*3+[0]*97
        process_test_list.append([feature,label])
        i=i+3
    #np.save("Data/processed_testset.npy", process_test_list)
    print("processed_testset saved")


process_rating()
create_testset()