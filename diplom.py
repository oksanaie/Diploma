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

def mean_average_precision_new(predicted_probability_distribution, 
                           list_of_classes, 
                           test_y):
    total_error = 0
    k = -1
    for row in predicted_probability_distribution:
        k += 1
        prob_of_class = 0
        place = 1
        for i in xrange(0, len(list_of_classes)):
            if list_of_classes[i] == test_y[k]:
                prob_of_class = row[i]
                break
        for i in xrange(0, len(list_of_classes)):
            if row[i] > prob_of_class:
                place += 1
        print place
        if place < 6:
            total_error += 1.0 / (place)
    return total_error / len(predicted_probability_distribution)
def mean_average_precision(predicted_probability_distribution, 
                           list_of_classes, 
                           test_y):
    total_error = 0
    k = -1
    for row in predicted_probability_distribution:
        k += 1
        classes_by_prob = []
        for i in xrange(0, len(list_of_classes)):
            classes_by_prob.append([row[i], list_of_classes[i]])
        classes_by_prob.sort(reverse=True)    
        for j in range (0, min(5, len(classes_by_prob))):
            if classes_by_prob[j][1] == test_y[k]:
                print j
                total_error += 1.0 / (j + 1)
    
    return total_error / len(predicted_probability_distribution)

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["random_forest", "k_neighbors", "logistic_regression"], default=["random_forest"], dest="model")
args = parser.parse_args()
print "Model used in this run is %s" % args.model

TRAIN_DATASET_SIZE = 10
TEST_DATASET_SIZE = 10

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
MAP_test_new = mean_average_precision_new(predicted_probability_distribution_test, 
                                  model.classes_, 
                                  test_y)
print "MAP new on test data: %.3f" % MAP_test_new
MAP_train = mean_average_precision(predicted_probability_distribution_train, 
                                   model.classes_, 
                                   train_y) 
print "MAP on train data: %.3f" % MAP_train
MAP_train_new = mean_average_precision_new(predicted_probability_distribution_train, 
                                   model.classes_, 
                                   train_y) 
print "MAP on new train data: %.3f" % MAP_train_new

print "Elapsed time: %.3f" % (time.time() - start_time)

