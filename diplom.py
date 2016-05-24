#!/usr/bin/python
import time
import ml_metrics as metrics
import argparse
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn import cross_validation
import numpy as np
from termcolor import colored
from sklearn.cross_validation import ShuffleSplit

def parse(line):
    tokens = line.split(',')
    features = [tokens[3], tokens[4], tokens[5], tokens[13], tokens[14], tokens[15], tokens[16], 
        tokens[17], tokens[20], tokens[21], tokens[22]]
    features = [float (f) for f in features]
    label = int (tokens[23])
    return (features, label)

def mean_average_precision(predicted_probability_distribution, 
                           list_of_classes, 
                           test_y):
    total_error = 0
    k = -1
    for row in predicted_probability_distribution:
        k += 1
        prob_of_target_class = 0
        place = 1
        assert len(list_of_classes) == len(row)
        for i in xrange(0, len(row)):
            if list_of_classes[i] == test_y[k]:
                prob_of_target_class = row[i]
                break
        for i in xrange(0, len(row)):
            if (row[i] > prob_of_target_class 
                or (row[i] == prob_of_target_class and list_of_classes[i] > test_y[k])):
                place += 1
                if place >= 6: break
        if place < 6:
            total_error += 1.0 / place
    return total_error / len(predicted_probability_distribution)

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
parser.add_argument("--model", choices=["random_forest", "k_neighbors", "logistic_regression"], default=["random_forest"], dest="model")
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
parser.add_argument("--test_dataset_size", default=0, type=int, dest="test_dataset_size")
args = parser.parse_args()
print "Model used in this run is %s" % colored(args.model, "red")
print "Train dataset size is %s" % colored(args.train_dataset_size, "red")
print "Test dataset size is %s" % colored(args.test_dataset_size, "red")

train = open ('train.csv', 'r')
train_X = []
train_y = []
test_X = []
test_y = []
cnt = -1
for line in train: 
    cnt += 1
    if cnt == 0:
        continue
    (features, label) = parse(line)
    if cnt <= args.train_dataset_size:
        train_X.append(features)
        train_y.append(label)
    elif cnt <= args.train_dataset_size + args.test_dataset_size:
        test_X.append(features)
        test_y.append(label)
    else: 
        break

print "Input/Output time: %.3f" % (time.time() - start_time)
train_start_time = time.time()

if args.model == "logistic_regression":
    model = linear_model.LogisticRegression(C=1, solver='lbfgs', 
                                            multi_class='multinomial', max_iter=100, n_jobs=-1)
elif args.model == "k_neighbors":
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=-1)

model.fit(train_X, train_y)

predicted_probability_distribution_test = model.predict_proba(test_X)
predicted_probability_distribution_train = model.predict_proba(train_X)

print "Training + prediction time: %.3f" % (time.time() - train_start_time)

MAP_test = mean_average_precision(predicted_probability_distribution_test, 
                                  model.classes_, 
                                  test_y)
print "MAP on test data: %.3f" % MAP_test
MAP_train = mean_average_precision(predicted_probability_distribution_train, 
                                   model.classes_, 
                                   train_y) 
print "MAP on train data: %.3f" % MAP_train

# plot_learning_curve(model, args.model, train_X, train_y, ylim=None, 
#     cv=ShuffleSplit(len(train_X), n_iter=1, test_size=.25, random_state=0),
#                     train_sizes=np.linspace(.1, 1.0, 5))
# plt.show()

print "Elapsed time: %.3f" % (time.time() - start_time)
