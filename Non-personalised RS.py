import numpy as np
import pandas as pd
from collections import Counter

def Non_prs():
    #load the dataset
    path='dataset/'
    title_r=['user_id','movie_id','rating','timestamp']
    title_m=['movie_id','title','genres']
    ratings= pd.read_table(path+"ratings.dat",sep='::',header=None,names=title_r,engine='python')
    movies= pd.read_table(path+"movies.dat",sep='::',header=None,names=title_m,engine='python',encoding="ISO-8859-1")

    #reindex the movie id
    old_id = movies["movie_id"].tolist()
    new_id = list(range(1, len(movies['movie_id']) + 1))
    map_dic = dict(zip(old_id, new_id))
    ratings["movie_id"] = ratings["movie_id"].apply(lambda x: map_dic[x])
    movies["movie_id"] = movies["movie_id"].apply(lambda x: map_dic[x])
    num_movie=max(movies["movie_id"])

    #define the weights for the viewed times and ratings
    viewed_weight=0.3
    ratings_weight=1-viewed_weight

    #the most viewed movies
    def Most_view_movies_list(ratings):
        viewed_list=Counter(list(ratings["movie_id"])).most_common(num_movie)
        Ranktable = list(map(list, viewed_list))
        Hotrank = [i[0] for i in Ranktable]
        return Hotrank

    #The highest rating list
    def The_highest_rating_list(ratings):
        movies_list=list((set(ratings["movie_id"].tolist())))
        movies_group = ratings.groupby(["movie_id"])
        movies_score=movies_group.mean()["rating"]
        movies_dict=dict(zip(movies_list,movies_score))
        rank_r = sorted(movies_dict.items(), key=lambda x: x[1], reverse=True)
        rank_rating=[i[0] for i in rank_r]
        return rank_rating

    #combine two rank list to caculate the overall rank
    HotRank=Most_view_movies_list(ratings)
    RatingRank=The_highest_rating_list(ratings)
    Overalltable=[[] for _ in range(num_movie+1)]
    OverScore=[[] for _ in range(num_movie+1)]
    for i in range(len(HotRank)): #HotRank[i]=movie_id
        Overalltable[HotRank[i]].append(i)
    for i in range(len(RatingRank)): #HotRank[i]=movie_id
        Overalltable[RatingRank[i]].append(i)
    for i in range(len(Overalltable)):
        if Overalltable[i]!=[]:
            OverScore[i]=[i,viewed_weight*Overalltable[i][0]+ratings_weight*Overalltable[i][1]]
    while [] in OverScore:
        OverScore.remove([])
    #
    OverallRank_o=sorted(OverScore,key=(lambda x:x[1]))
    OverallRank=list(map(list, zip(*OverallRank_o)))[0]
    for i in range(20):
        print(movies[movies["movie_id"]==OverallRank[i]]["title"].to_string())

    #save the OverallRank data
    np.save("Data/OverallRank1.npy", OverallRank)
    print("saved rank")
    return OverallRank


#evaluate(Hit_rate, MAE, NDCG)
def evaluate_model(OverallRank, test_set, K):
    def hit_check(true_instance, ranklist):
        if true_instance in ranklist:
            return 1
        return 0

    def MAE(true_instance, prediction_list_sorted):
        index = prediction_list_sorted.index(true_instance)
        l=len(prediction_list_sorted)
        mark = index/l
        error = abs(float(mark) - 1)
        return error

    def NDCG(true_instance, prediction_list_sorted): #Normalize Discounted Cumulative Gain
        index = prediction_list_sorted.index(true_instance)
        return np.log(2) / np.log(index + 2)

    hit_times = []
    MAE_list=[]
    NDCG_list=[]
    for i in range(len(test_set)):
        feature_list = list(map(lambda x: list(x), test_set[i][0]))
        true_instance = feature_list[0][1]
        ranklist = OverallRank
        hit_times.append(hit_check(true_instance, ranklist[:10]))
        MAE_list.append(MAE(true_instance,ranklist))
        NDCG_list.append(NDCG(true_instance, ranklist))


    return np.mean(hit_times),np.mean(MAE_list),np.mean(NDCG_list)

#this code is used to save the overall rank
OverallRank=Non_prs()
test_set=np.load('Data/processed_testset.npy', allow_pickle=True)
OverallRank=np.load('Data/OverallRank.npy',allow_pickle=True)
OverallRank=OverallRank.tolist()
hit_rate,Mae,ndcg=evaluate_model(OverallRank, test_set, 10)
print(("\nNon-personalise recommendation MAE = %.3f, hr = %.3f, ndcg = %3f") % (Mae, hit_rate, ndcg))
