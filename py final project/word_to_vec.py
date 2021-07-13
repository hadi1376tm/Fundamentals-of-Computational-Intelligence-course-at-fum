import math

import gensim.downloader as api
import numpy as np
import pandas as pd

model = api.load("glove-wiki-gigaword-50")

train_data = pd.read_pickle("text_preprocessing_train_pkl")
test_data = pd.read_pickle("text_preprocessing_test_pkl")


def avg(temp):
    my_sum = np.zeros(50)
    count = 0
    for word in temp:
        try:
            my_sum = my_sum + model.get_vector(word)
            count += 1
        except Exception as e:
            pass
    return my_sum / count


def array_to_int(temp):
    my_sum = 0
    for i in range(20):
        if temp[i] == 1:
            my_sum += math.pow(2, i)
    return my_sum


train_data['overview'] = train_data['overview'].apply(avg)
train_data = train_data[train_data.overview.apply(lambda x: np.isnan(x).sum()) == 0] #dumps nulls

test_data['overview'] = test_data['overview'].apply(avg)
test_data = test_data[test_data.overview.apply(lambda x: np.isnan(x).sum()) == 0]

train_data['geners'] = np.apply_along_axis(array_to_int, 1, train_data.drop(columns=["overview"]).to_numpy()) # turn included genres to a number usin to_numpy
test_data['geners'] = np.apply_along_axis(array_to_int, 1, test_data.drop(columns=["overview"]).to_numpy())
print(train_data[['overview', 'geners']])
print(test_data[['overview', 'geners']])

train_data[['overview', 'geners']].to_pickle("final_train_pkl")
test_data[['overview', 'geners']].to_pickle("final_test_pkl")

