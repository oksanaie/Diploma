#!/usr/bin/python
import time
import numpy as np
from termcolor import colored
import pickle
import common
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", default=model.pkl, dest="model_filename")
#parser.add_argument("--test_dataset_size", default=700000, type=int, dest="test_dataset_size")
args = parser.parse_args()
print "Test dataset size is %s" % colored(args.test_dataset_size, "red")

test_X, _unused = load_dataset_from_file('test.csv', 700000, False)

model_file = open(args.model_filename, 'rb')
model = pickle.load(model_file)
model_file.close()

predicted_probability_distribution_test = model.predict_proba(test_X)

result = []
result.append(["id", "hotel_cluster"])
k = -1
for row in predicted_probability_distribution:
    k += 1
    line = []
    line.append(k)
    classes_by_prob = []
    for i in xrange(0, len(row)):
        classes_by_prob.append([row[i], model.classes_[i]])
        classes_by_prob.sort(reverse=True)
    clusters = [classes_by_prob[x][1] for x in xrange(0, 5)]  
    #clusters =[] 
    # for x in xrange(0, 5):
    #     clusters.append(classes_by_prob[x][1])
    line.append(clusters)
    result.append(line)

with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(result)