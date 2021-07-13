from scipy.stats import mode
import numpy as np
#from mnist import MNIST
from time import time
import pandas as pd
import os
import matplotlib.pyplot as matplot
import matplotlib
#%matplotlib inline

import random
matplot.rcdefaults()
from time import time
from IPython.display import display, HTML
from itertools import chain
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')
import input_data
mnist = input_data.read_data_sets("samples/", one_hot=True)
train = mnist.train.images
validation = mnist.validation.images
test = mnist.test.images
print("55555555555")
trlab = mnist.train.labels
vallab = mnist.validation.labels
tslab = mnist.test.labels

train = np.concatenate((train, validation), axis=0)
trlab = np.concatenate((trlab, vallab), axis=0)
mlp = MLPClassifier()
mlp.fit(train, trlab)
accuracy_score(tslab, mlp.predict(test)) # Test Accuracy
i = 0
df = pd.DataFrame(columns=['alpha', 'max_iter', 'train_acc', 'test_acc', 'train_time'])
for a in [  1]:
    for mi in [200]:
        st = time()
        mlp = MLPClassifier(alpha=a, max_iter=mi)
        mlp.fit(train, trlab)
        end = time() - st

        acc_tr = accuracy_score(trlab, mlp.predict(train))  # Train Accuracy
        acc = accuracy_score(tslab, mlp.predict(test))  # Test Accuracy
        df.loc[i] = [a, mi, acc_tr, acc, end]
        i = i + 1

df # Results
acc = []
acc_tr = []
timelog = []
for l in [10, 20, 50, 100, 200, 500, 1000]:
    t = time()
    mlp = MLPClassifier(alpha=0.1, max_iter=200, hidden_layer_sizes=(l,))
    mlp.fit(train, trlab)
    endt = time() - t

    a_tr = accuracy_score(trlab, mlp.predict(train))  # Train Accuracy
    a = accuracy_score(tslab, mlp.predict(test))  # Test Accuracy

    acc_tr.append(a_tr)
    acc.append(a)
    timelog.append(endt)

l = [10,20,50,100,200,500,1000]
N = len(l)
l2 = np.arange(N)
matplot.subplots(figsize=(10, 5))
matplot.plot(l2, acc, label="Testing Accuracy")
matplot.plot(l2, acc_tr, label="Training Accuracy")
matplot.xticks(l2,l)
matplot.grid(True)
matplot.xlabel("Hidden Layer Nodes")
matplot.ylabel("Accuracy")
matplot.legend()
matplot.title('Accuracy versus Nodes in the Hidden Layer for MLPClassifier', fontsize=12)
matplot.show()
l = [10,20,50,100,200,500,1000]
N = len(l)
l2 = np.arange(N)
matplot.subplots(figsize=(10, 5))
matplot.plot(l2, timelog, label="Training time in s")
matplot.xticks(l2,l)
matplot.grid(True)
matplot.xlabel("Hidden Layer Nodes")
matplot.ylabel("Time (s)")
matplot.legend()
matplot.title('Training Time versus Nodes in the Hidden Layer for MLPClassifier', fontsize=12)
matplot.show()

