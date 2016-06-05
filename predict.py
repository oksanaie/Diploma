#!/usr/bin/python
import time
import numpy as np
from termcolor import colored
import pickle
from common import load_dataset_from_file
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", dest="model_filename", default="model")
args = parser.parse_args()

test_X, _unused = load_dataset_from_file('test.csv', 3000001, False)

model = None
with open(args.model_filename, 'rb') as model_file:
    model = pickle.load(model_file)
    model_file.close()

print "Trying to predict probability distributions."
predicted_probability_distribution_test = model.predict_proba(test_X)
print "Done"
result = []
result.append(["id", "hotel_cluster"])
k = -1
for row in predicted_probability_distribution_test:
    k += 1
    line = []
    line.append(k)
    classes_by_prob = []
    for i in xrange(0, len(row)):
        classes_by_prob.append([row[i], model.classes_[i]])
    classes_by_prob.sort(reverse=True) 
    line.append(" ".join([str(classes_by_prob[x][1]) for x in xrange(0, 5)]))
    if (k + 1) % 100000 == 0:
        print "Predicted %d examples." % (k + 1)
    result.append(line)

with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(result)
