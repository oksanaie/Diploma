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

from sklearn import cross_validation
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

allX, ally = load_dataset_from_file('train.csv', args.train_dataset_size, True)

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(
# 	allX, ally, test_size=0.3, random_state=0)

# # n_estimator = 10, depth = 20
# model = FastRandomForest(n_jobs=-1)
# model.fit(X_train, y_train)

# print model.score(X_test, y_test)

def plot_n_estimators(train_X, train_y):
	n_estimators = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
	param_grid = dict(n_estimators=n_estimators)

	grid = GridSearchCV(estimator=FastRandomForest(n_jobs=-1), param_grid=param_grid)
	grid.fit(train_X, train_y)

	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(n_estimators))

	plt.title('n_estimators')
	plt.plot(n_estimators, scores)
	plt.show()

def plot_max_depth(train_X, train_y):
	param = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
	grid = GridSearchCV(
		estimator=FastRandomForest(n_jobs=-1), 
		param_grid=dict(max_depth=param))
	grid.fit(train_X, train_y)

	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(param))

	plt.title('max_depth')
	plt.plot(param, scores)
	plt.show()

def plot_max_features(train_X, train_y):
	param = range(1, 30)
	grid = GridSearchCV(
		estimator=FastRandomForest(n_jobs=-1), 
		param_grid=dict(max_features=param))
	grid.fit(train_X, train_y)

	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(param))

	plt.title('max_features')
	plt.plot(param, scores)
	plt.show()

#plot_n_estimators(allX, ally)
#plot_max_depth(allX, ally)
plot_max_features(allX, ally)
