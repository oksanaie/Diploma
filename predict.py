#!/usr/bin/python
import time
import numpy as np
from termcolor import colored
import pickle
import common.py

parser = argparse.ArgumentParser()
parser.add_argument("--test_dataset_size", default=10000, type=int, dest="test_dataset_size")
args = parser.parse_args()
print "Test dataset size is %s" % colored(args.test_dataset_size, "red")

test_X, test_y = load_dataset_from_file('test.csv', args.test_dataset_size, False)

model_file = open('model.pkl', 'rb')
model = pickle.load(model_file)
model_file.close()

predicted_probability_distribution_test = model.predict_proba(test_X)
