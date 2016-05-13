#!/usr/bin/python
print "hello"

def parse(line):
    tokens = line.split(',')
    features = [tokens[3], tokens[4], tokens[5], tokens[13], tokens[14], tokens[15], tokens[16], 
        tokens[17], tokens[20], tokens[21], tokens[22]]
    features = [float (f) for f in features]
    label = int (tokens[23])
    return (features, label)

def mean_average_precision (predicted_probability_distribution, 
    n_of_classes, 
    list_of_classes, 
    test_y, 
    dataset_size):
    AP = 0
    k = -1
    for row in predicted_probability_distribution:
        k += 1
        classes_by_prob = []
        for i in xrange(0, n_of_classes):
            classes_by_prob.append([row[i], list_of_classes[i]])
            classes_by_prob.sort(reverse=True)    
        for j in range (0, 5):
            if classes_by_prob[j][1] == test_y[k]:
                AP += 1.0 / (j+1)
    MAP = AP / dataset_size
    return MAP

TRAIN_DATASET_SIZE = 30000
TEST_DATASET_SIZE = 30000

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
#
# print train_X[0], train_y[0]
# print test_X[0], test_y[0]

from sklearn import linear_model
logistic = linear_model.LogisticRegression(C = 1, solver = 'lbfgs', 
    multi_class = 'multinomial', max_iter = 100)
print('LogisticRegression score: %f'
      % logistic.fit(train_X, train_y).score(test_X, test_y))
predicted_probability_distribution = logistic.predict_proba(test_X)
n_of_classes = len(logistic.classes_)
list_of_classes = logistic.classes_
MAP = mean_average_precision (predicted_probability_distribution, 
    n_of_classes, 
    list_of_classes, 
    test_y, 
    TEST_DATASET_SIZE)

print MAP 
# from sklearn import svm
# clf = svm.SVC()
# clf.fit (r, b)

# print "hello"

# # head -n1500 train.csv > minitest.txt


# #print (tr)
# print(clf.predict(tr))
# print (ta)
