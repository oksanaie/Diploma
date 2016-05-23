#!/usr/bin/python
import time
import ml_metrics as metrics
import argparse
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    for k, row in enumerate (predicted_probability_distribution):
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
        if place < 6:
            total_error += 1.0 / place
    return total_error / len(predicted_probability_distribution)

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["random_forest", "k_neighbors", "logistic_regression"], default=["random_forest"], dest="model")
parser.add_argument("--dataset_size", default=100000, type=int, dest="dataset_size")
args = parser.parse_args()
print "Model used in this run is %s" % args.model
print "Dataset size is %s" % args.dataset_size

TRAIN_DATASET_SIZE = args.dataset_size
TEST_DATASET_SIZE = args.dataset_size

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
    if cnt <= TRAIN_DATASET_SIZE:
        train_X.append(features)
        train_y.append(label)
    elif cnt <= TRAIN_DATASET_SIZE + TEST_DATASET_SIZE:
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
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1)

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

print "Elapsed time: %.3f" % (time.time() - start_time)

