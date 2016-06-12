#!/usr/bin/python
# To prepare a file for submission one needs to run this script first.
# Then ./predict.py and ./leak.py. Resulting file 'final_predictions.csv'
# is good for the submission.
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

from termcolor import colored

from sklearn.cross_validation import ShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

from common import load_dataset_from_file
from common import FastRandomForest

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")

args = parser.parse_args()
train_X, train_y = load_dataset_from_file('train.csv', args.train_dataset_size, True)

n_estimators = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
param_grid = dict(n_estimators=n_estimators)

grid = GridSearchCV(estimator=FastRandomForest(n_jobs=-1), param_grid=param_grid)
grid.fit(train_X, train_y)

scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(n_estimators))

plt.title('n_estimators')
plt.plot(n_estimators, scores)
plt.show()
