#!/usr/bin/python
import time
import numpy as np
from termcolor import colored
import pickle
from common import load_dataset_from_file
from common import parse
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", dest="model_filename")
args = parser.parse_args()

test_X, _unused = load_dataset_from_file('test.csv', 7000000, False)

model = None
with open(args.model_filename, 'rb') as model_file:
    model = pickle.load(model_file)
    model_file.close()

predicted_probability_distribution_test = model.predict_proba(test_X)

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
    if len(result) % 100000 == 0:
        print classes_by_prob
        print "Processed %d rows, predicted %d examples." % (k, len(result))
    result.append(line)

with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(result)
