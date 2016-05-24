#!/usr/bin/python
import time
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from termcolor import colored
import pickle
import common.py

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--model_filename", default=model.pkl, dest="model_filename")
parser.add_argument("--model", choices=["random_forest", "k_neighbors", "logistic_regression"], default=["random_forest"], dest="model")
parser.add_argument("--train_dataset_size", default=100000, type=int, dest="train_dataset_size")
args = parser.parse_args()
print "Model used in this run is %s" % colored(args.model, "red")
print "Train dataset size is %s" % colored(args.train_dataset_size, "red")

train_X, train_y = load_dataset_from_file('train.csv', args.train_dataset_size, True)

print "Input/Output time: %.3f" % (time.time() - start_time)
train_start_time = time.time()

if args.model == "logistic_regression":
    model = linear_model.LogisticRegression(C=1, solver='lbfgs', 
                                            multi_class='multinomial', max_iter=100, n_jobs=-1)
elif args.model == "k_neighbors":
    model = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
else:
    model = RandomForestClassifier(n_estimators=10, max_depth=20, n_jobs=-1)

my_model = model.fit(train_X, train_y)
model_file = open(args.model_filename, 'wb')
pickle.dump(my_model, model_file)
model_file.close()
print "Elapsed time: %.3f" % (time.time() - start_time)
