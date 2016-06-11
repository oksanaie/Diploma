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

max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
param_grid = dict(max_depth=max_depth)

grid = GridSearchCV(estimator=FastRandomForest(n_jobs=-1), param_grid=param_grid)
grid.fit(train_X, train_y)

scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(max_depth))

plt.title('max_depth')
plt.plot(max_depth, scores)
plt.show()
