#!/usr/bin/python
print "hello"

def parse(line):
    tokens = line.split(',')
    features = [tokens[3], tokens[4], tokens[5], tokens[13], tokens[14], tokens[15], tokens[16], 
        tokens[17], tokens[20], tokens[21], tokens[22]]
    features = [float (f) for f in features]
    label = int (tokens[23])
    return (features, label)

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

# print train_X[0], train_y[0]
# print test_X[0], test_y[0]

from sklearn import linear_model
logistic = linear_model.LogisticRegression(C = 2, solver = 'newton-cg', 
    multi_class = 'ovr', max_iter = 10000)
print('LogisticRegression score: %f'
      % logistic.fit(train_X, train_y).score(test_X, test_y))
Predicted_probability_distribution = logistic.predict_proba(test_X)
#print logistic.classes_
AP = 0
k = -1
for row in Predicted_probability_distribution:
    k += 1
    r =[]
    for i in xrange(0, len(logistic.classes_)):
        r.append([row[i], logistic.classes_[i]])
    r.sort(reverse=True)    
    for j in range (0, 5):
        if r[j][1] == test_y[k]:
            AP += 1.0/(j+1)
MAP = AP / TEST_DATASET_SIZE
print (MAP)
# from sklearn import svm
# clf = svm.SVC()
# clf.fit (r, b)

# print "hello"

# # head -n1500 train.csv > minitest.txt
# z = open ('minitest.txt', 'r')
# c = []
# for line in z:
#   c.append(line)
# t = []
# for i in xrange(1501, len(c)):
#   t.append(c[i])
# for i in range(len(t)):
#   t[i] = t[i].split(',')
# tr = []
# for i in t:
#   tr.append([i[13], i[14], i[15], i[16], i[17], i[18], i[19], i[20], i[21], i[22]])
# ta = []
# for i in t:
#   ta.append(i[23])

# #print (tr)
# print(clf.predict(tr))
# print (ta)
