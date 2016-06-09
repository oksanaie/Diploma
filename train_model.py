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

from common import load_dataset_from_file
from common import FastRandomForest

def plot_learning_curve(estimator, title, X, y, curve_points=5):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
#        train_sizes=np.linspace(.1, 1.0, curve_points),
        train_sizes=np.logspace(0.0, 4.0, curve_points, base=1.0/3.0),
        n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, 
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, 
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, 
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, 
                     alpha=0.1, 
                     color="g")
    plt.plot(train_sizes, 
             train_scores_mean, 'o-', 
             color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, test_scores_mean[-1], test_scores_std[-1]

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", dest="model_filename", default="model")
parser.add_argument(
    "--model", 
    choices=["random_forest", "k_neighbors", "logistic_regression"], 
    default=["random_forest"], 
    dest="model")
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
parser.add_argument("--plot_learning_curve", default=False, type=bool, dest="plot_learning_curve")
parser.add_argument("--curve_points", default=5, type=int, dest="curve_points")

args = parser.parse_args()
print "Model used in this run is %s" % colored(args.model, "red")
print "Train dataset size is %s" % colored(args.train_dataset_size, "red")

train_X, train_y = load_dataset_from_file('train.csv', args.train_dataset_size, True)

print "Input/Output time: %.3f" % (time.time() - start_time)
train_start_time = time.time()

if args.model == "logistic_regression":
    model = LogisticRegression(C=0.1, 
                               solver='lbfgs', 
                               multi_class='multinomial', 
                               max_iter=100)
elif args.model == "k_neighbors":
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
else:
    model = FastRandomForest(n_estimators=30, max_depth=20, n_jobs=-1)

if args.plot_learning_curve:
    print "Plotting learning curve."
    plt, test_map_avg, test_map_std = plot_learning_curve(model, args.model, train_X, train_y, args.curve_points)
    print "Mean Average Precision on test: %.3f, 95%% confidence [%.3f, %.3f]" % (
        test_map_avg, 
        test_map_avg - 2. * test_map_std, 
        test_map_avg + 2. * test_map_std)
    plt.show()
else:
    print "Training model."
    model.fit(train_X, train_y)
    # print "Feature importances: "
    # for x in xrange (0, len(model.feature_importances_)):
    #     print x, model.feature_importances_[x]
    print "Saving model to %s." % args.model_filename
    with open(args.model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

print "Elapsed time: %.3f" % (time.time() - start_time)
