#!/usr/bin/python
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from termcolor import colored
import pickle
from common import load_dataset_from_file
from common import parse
import numpy as np
import argparse

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", dest="model_filename")
parser.add_argument("--model", choices=["random_forest", "k_neighbors", "logistic_regression"], default=["random_forest"], dest="model")
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
parser.add_argument("--plot_learning_curve", default=False, type=bool, dest="plot_learning_curve")
args = parser.parse_args()
print "Model used in this run is %s" % colored(args.model, "red")
print "Train dataset size is %s" % colored(args.train_dataset_size, "red")

train_X, train_y = load_dataset_from_file('train.csv', args.train_dataset_size, True)

print "Input/Output time: %.3f" % (time.time() - start_time)
train_start_time = time.time()

if args.model == "logistic_regression":
    model = LogisticRegression(C=1, solver='lbfgs', 
                                            multi_class='multinomial', max_iter=100, n_jobs=-1)
elif args.model == "k_neighbors":
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=-1)

if args.plot_learning_curve:
    print "Plotting learning curve."
    plt = plot_learning_curve(
        model, 
        args.model, 
        train_X,
        train_y,
        cv=ShuffleSplit(len(train_X), n_iter=1, test_size=.25, random_state=0))
    plt.show()

print "Training model."
my_model = model.fit(train_X, train_y)
model_file = open(args.model_filename, 'wb')
pickle.dump(my_model, model_file)
model_file.close()

print "Elapsed time: %.3f" % (time.time() - start_time)
