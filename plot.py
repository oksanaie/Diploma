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
from common import FastLogisticRegression

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
args = parser.parse_args()

allX, catX, ally = load_dataset_from_file('train.csv', args.train_dataset_size, True)

DIR = "illustrations"
CROSS_VALIDATION = cross_validation.ShuffleSplit(len(ally), n_iter=1, test_size=0.30, random_state=0)

LOG_REGRESSION = FastLogisticRegression(C=1,
                                        max_iter=30, 
                                        n_jobs=-1)
KNN = CustomKNN(n_neighbors=10, n_jobs=-1)
RANDOM_FOREST = FastRandomForest(n_estimators=100, 
                                 max_depth=15, 
                                 max_features=10, 
                                 n_jobs=-1)

def plot_n_estimators(train_X, train_y):
    print "plot_n_estimators()"
    param = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastRandomForest(n_jobs=-1),
        param_grid=dict(n_estimators=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(param))
    plt.plot(param, scores)

def plot_max_depth(train_X, train_y):
    print "plot_max_depth()"
    param = [5, 10, 15, 20, 25, 30, 50, 75, 100, 150, 200]
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastRandomForest(n_jobs=-1), 
        param_grid=dict(max_depth=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(param))
    plt.plot(param, scores)

def plot_n_estimators_vs_max_depth(train_X, train_y):
    print "plot_n_estimators_vs_max_depth()"
    n_estimators = [25, 50, 75, 100, 150, 200]
    max_depth = [5, 10, 15, 20, 25, 30]
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastRandomForest(n_jobs=-1),
        param_grid=dict(n_estimators=n_estimators, max_depth=max_depth))

    grid.fit(train_X, train_y)

    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(n_estimators), len(max_depth))

    plt.figure(figsize=(8, 6))
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('n_estimators')
    plt.ylabel('max_depth')
    plt.colorbar()
    plt.xticks(np.arange(len(n_estimators)), n_estimators, rotation=45)
    plt.yticks(np.arange(len(max_depth)), max_depth)
    plt.title('Validation accuracy')

def plot_max_features(train_X, train_y):
    print "plot_max_features()"
    param = range(1, 30, 2)
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastRandomForest(n_jobs=-1), 
        param_grid=dict(max_features=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(param))
    plt.plot(param, scores)

def plot_n_neighbours(train_X, train_y, weights):
    print "plot_n_neighbours(): %s" % weights
    param = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=CustomKNN(n_jobs=-1, weights=weights), 
        param_grid=dict(n_neighbors=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    np.array(scores).reshape(len(param))
    line, = plt.plot(param, scores, label=weights)
    plt.legend(loc=1)

def plot_learning_curve(train_X, train_y, estimator, title, 
                        n_jobs=-1, curve_points=5, color_a='r', color_b='g'):
    print "plot_learning_curve(): %s" % title
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        train_X,
        train_y,
        cv=CROSS_VALIDATION,
        train_sizes=np.linspace(.1, 1.0, curve_points),
        n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    if color_a:
        plt.fill_between(train_sizes, 
                         train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, 
                         alpha=0.1,
                         color=color_a)
        plt.plot(train_sizes, 
                 train_scores_mean, 
                 'o-', 
                 color=color_a,
                 label="Training score (%s)" % title)
    if color_b:
        plt.fill_between(train_sizes, 
                         test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, 
                         alpha=0.1, 
                         color=color_b)
        plt.plot(train_sizes, 
                 test_scores_mean, 
                 'o-', 
                 color=color_b,
                 label="Cross-validation score (%s)" % title)

    plt.legend(loc="best", prop={'size':10})
    return plt, test_scores_mean[-1], test_scores_std[-1]

def plot_regularization(train_X, train_y):
    print "plot_regularization()"
    param = [1e-5 * 3 ** pw for pw in xrange(0, 21, 2)]
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastLogisticRegression(n_jobs=-1, max_iter=30), 
        param_grid=dict(C=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(param))
    plt.plot(xrange(0, 21, 2), scores)

def plot_max_iter(train_X, train_y):
    print "plot_max_iter()"
    param = range(0, 100, 10)
    grid = GridSearchCV(
        cv=CROSS_VALIDATION,
        estimator=FastLogisticRegression(n_jobs=-1), 
        param_grid=dict(max_iter=param))
    grid.fit(train_X, train_y)
    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(param))
    plt.plot(param, scores)

def save_and_clear(title):
    print "save_and_clear(): %s" % title
    plt.title(title)
    plt.savefig(os.path.join(DIR, title + '.png'))
    plt.close()

######## RANDOM FOREST #########
# plot_n_estimators(allX, ally)
# save_and_clear('rf.n_estimators')

# plot_max_depth(allX, ally)
# save_and_clear('rf.max_depth')

# plot_max_features(allX, ally)
# save_and_clear('rf.max_features')

# plot_n_estimators_vs_max_depth(allX, ally)
# save_and_clear('rf.n_estimators_vs_max_depth')

# plot_learning_curve(allX, ally, RANDOM_FOREST, 'Random Forest')
# save_and_clear('rf.learning_curve')

# # ######### KNN ###################
# plot_n_neighbours(catX, ally, 'uniform')
# plot_n_neighbours(catX, ally, 'distance')
# save_and_clear('knn.n_neighbors')

# plot_learning_curve(catX, ally, KNN, 'K Nearest Neighbors')
# save_and_clear('knn.learning_curve')

# ######### LOGISTIC REGRESSION #####
# plot_regularization(catX, ally)
# save_and_clear('log_reg.c')

# plot_max_iter(catX, ally)
# save_and_clear('log_reg.max_iter')

# plot_learning_curve(catX, ally, LOG_REGRESSION, 'Logistic Regression', n_jobs=1)
# save_and_clear('log_reg.learning_curve')

########## COMBINED #############
plot_learning_curve(catX, ally, LOG_REGRESSION, 'Logistic Regression', n_jobs=1, color_a=None, color_b='r')
plot_learning_curve(catX, ally, KNN, 'K Nearest Neighbors', color_a=None, color_b='g')
plot_learning_curve(allX, ally, RANDOM_FOREST, 'Random Forest', color_a=None, color_b='b')
save_and_clear('learning_curves')
