#!/usr/bin/python
import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import os

from termcolor import colored

from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

from common import load_dataset_from_file
from common import FastRandomForest
from common import CustomKNN

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
args = parser.parse_args()

allX, ally = load_dataset_from_file('train.csv', args.train_dataset_size, True)

DIR = "illustrations"

def plot_n_estimators(train_X, train_y):
	param = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
	grid = GridSearchCV(
		estimator=FastRandomForest(n_jobs=-1), 
		param_grid=dict(n_estimators=param))
	grid.fit(train_X, train_y)
	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(param))
	plt.plot(param, scores)

def plot_max_depth(train_X, train_y):
	param = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
	grid = GridSearchCV(
		estimator=FastRandomForest(n_jobs=-1), 
		param_grid=dict(max_depth=param))
	grid.fit(train_X, train_y)
	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(param))
	plt.plot(param, scores)

def plot_max_features(train_X, train_y):
	param = range(1, 30, 2)
	grid = GridSearchCV(
		estimator=FastRandomForest(n_jobs=-1), 
		param_grid=dict(max_features=param))
	grid.fit(train_X, train_y)
	scores = [x[1] for x in grid.grid_scores_]
	scores = np.array(scores).reshape(len(param))
	plt.plot(param, scores)

def plot_n_neighbours(train_X, train_y, weights):
	param = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
	grid = GridSearchCV(
		estimator=CustomKNN(n_jobs=-1, weights=weights), 
		param_grid=dict(n_neighbors=param))
	grid.fit(train_X, train_y)
	scores = [x[1] for x in grid.grid_scores_]
	np.array(scores).reshape(len(param))
	line, = plt.plot(param, scores, label=weights)
	plt.legend(loc=1)

def save_and_clear(title):
	plt.title(title)
	plt.savefig(os.path.join(DIR, title + '.png'))
	plt.close()

#plot_n_estimators(allX, ally)
#save_and_clear('n_estimators')
#plot_max_depth(allX, ally)
#save_and_clear('max_depth')
#plot_max_features(allX, ally)
#save_and_clear('max_features')
plot_n_neighbours(allX, ally, 'uniform')
plot_n_neighbours(allX, ally, 'distance')
save_and_clear('n_neighbors')
