import ast

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


from Text_Preprocessing import text_preprocessing

train_df = pd.read_csv("train.csv", encoding='latin1')
test_df = pd.read_csv("test.csv", encoding='latin1')


def convert_genres(x):
    x = ast.literal_eval(x)
    gener_ids = []
    for item in x:
        gener_ids.append(dict(item)['name'])
    return gener_ids


train_df = train_df[['genres', 'overview']]
test_df = test_df[['genres', 'overview']]
train_df['genres'] = train_df['genres'].apply(convert_genres)
test_df['genres'] = test_df['genres'].apply(convert_genres)

train_df['overview'] = train_df['overview'].astype(str)
test_df['overview'] = test_df['overview'].astype(str)

mlb = MultiLabelBinarizer()
train_df = train_df.join(pd.DataFrame(mlb.fit_transform(train_df.pop('genres')),
                                      columns=mlb.classes_,
                                      index=train_df.index))
test_df = test_df.join(pd.DataFrame(mlb.transform(test_df.pop('genres')),
                                    columns=mlb.classes_,
                                    index=test_df.index))

train_df = train_df[train_df['overview'].apply(len) > 10]
test_df = test_df[test_df['overview'].apply(len) > 10]
train_df['overview'] = train_df['overview'].apply(text_preprocessing)
test_df['overview'] = test_df['overview'].apply(text_preprocessing)
print(train_df['overview'])
print(test_df['overview'])
test_df.to_pickle("text_preprocessing_test_pkl")
train_df.to_pickle("text_preprocessing_train_pkl")

