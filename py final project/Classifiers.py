import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_pickle("final_train_pkl")
test_data = pd.read_pickle("final_test_pkl")

temp_minmax = MinMaxScaler()

x_train = np.array(train_data['overview'].values.tolist())
x_train = temp_minmax.fit_transform(x_train) #scaling to range [0,1]
y_train = train_data['geners'].to_numpy().reshape(-1, 1)  # to 2d array

x_test = np.array(test_data['overview'].values.tolist())
x_test = temp_minmax.transform(x_test)
y_test = test_data['geners'].to_numpy().reshape(-1, 1)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print("accuracy:")
print(accuracy_score(y_test, y_predict))
