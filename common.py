import datetime
import numpy as np

from features import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

N_COMPONENTS = 10 # MAX 149

def load_destinations():
    result = {}
    data = open ('destinations.csv', 'r').readlines()
    for line in data[1:]:
        tokens = line.split(',')
        result[int(tokens[0])] = [float(x) for x in tokens[1:]]

    X = np.array([result[x] for x in result.keys()])
    pca = decomposition.PCA(n_components=N_COMPONENTS)
    pca.fit(X)
    X = pca.transform(X)
    for i, x in enumerate(result.keys()):
        result[x] = X[i].tolist()
        assert len(result[x]) == N_COMPONENTS
    return result

DESTINATIONS = load_destinations()

def get_destination_features(srch_destination_id, feature_num=N_COMPONENTS):
    default_destination = [0.0 for i in xrange(0, feature_num)]
    return DESTINATIONS.get(srch_destination_id, default_destination)[0:feature_num]

def date(dt_str):
    try:
        return datetime.datetime.strptime(dt_str, "%Y-%m-%d")  
    except ValueError as e:
        print "Non-fatal error: %s" % e
        return datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")  

def parse_line(line, is_labeled, is_header=False):
    if line[-1] == '\n':
        line = line[:-1]
    tokens = line.split(",")
    if not is_header and is_labeled and int(tokens[18]) == 0:
        return (None, None)
    # Removing 'id'(0) from test
    if not is_labeled:
        tokens = tokens[1:]
    # Removing 'is_booking'(18), 'cnt'(19) fields that
    # are not available in test.
    else:
        tokens = tokens[0:18] + tokens[20:]
    # Getting label and deleting it from tokens
    label = None
    if is_labeled:
        label = tokens[21]
        tokens = tokens[:21] + tokens[22:]
    return (tokens, label)

CATEGORICAL_FEATURES = {1, 7, 8, 10, 11, 15, 18, 19}

def get_features(line, is_labeled=True):
    (tokens, label) = parse_line(line, is_labeled)
    if tokens is None:
        return None
    # One should use in this function only constants defined in features.py
    # like SRCH_CI, SRCH_CO, USER_LOCATION_COUNTRY, ... Please avoid 
    # referencing tokens directly as this is harmful for readability and
    # is error-prone.
    check_in = date(tokens[SRCH_CI])
    check_out = date(tokens[SRCH_CO])
    len_of_stay = int(check_out.strftime('%j')) - int(check_in.strftime('%j'))
    if len_of_stay < 0:
       len_of_stay += 365
    weekends = 0
    if (check_in.weekday() == 4 or check_in.weekday() == 5) and (len_of_stay < 4):
        weekends = 1
    # These are features that we will use as is:
    RAW_FEATURES = [
        SITE_NAME, # 0 
        POSA_CONTINENT, # 1 
        USER_LOCATION_COUNTRY, # 2
        USER_LOCATION_REGION, # 3
        USER_LOCATION_CITY, # 4
        ORIG_DESTINATION_DISTANCE, # 5
        USER_ID, # 6
        IS_MOBILE, # 7
        IS_PACKAGE, # 8
        CHANNEL, # 9
        SRCH_ADULTS_CNT, # 10
        SRCH_CHILDREN_CNT, # 11
        SRCH_RM_CNT, # 12
        SRCH_DESTINATION_ID, # 13
        SRCH_DESTINATION_TYPE_ID, # 14
        HOTEL_CONTINENT, # 15
        HOTEL_COUNTRY, # 16
        HOTEL_MARKET] # 17
    extra_features = [
        len_of_stay, # 18
        weekends, # 19
        int(check_in.strftime('%j')), # 20
        int(check_out.strftime('%j')), # 21
        int(check_in.month), # 22
        int(check_out.month), # 23
        int(check_in.strftime('%W')), # 24 
        int(check_out.strftime('%W'))] # 25
    extra_features += get_destination_features(int(tokens[SRCH_DESTINATION_ID])) # 26-35

    features = [
        tokens[raw_feature_id]
        for raw_feature_id in RAW_FEATURES] + extra_features

    final_features = []
    for i, f in enumerate(features):
        if i in CATEGORICAL_FEATURES:
            final_features.append(int (f) if f != '' else 0)
        else:
            final_features.append(float (f) if f != '' else 0.0)
    assert len(final_features) == 36
    return (final_features, float(label))


def load_dataset_from_file(filename, examples_count, is_labeled=True, expand_categorical=True):
    data = open (filename, 'r').readlines()
    # Next two lines verifies that the parsing result of header is what
    # we expect.
    header, _unused = parse_line(data[0], is_labeled, is_header=True)
    assert header == EXPECTED_HEADER
    data_X = []
    data_y = []
    cnt = 0
    for line in data[1:]:
        cnt += 1
        if len(data_X) == examples_count:
            break
        parse_result = get_features(line, is_labeled)
        if parse_result == None:
            continue
        (features, label) = parse_result
        data_X.append(np.array(features))
        data_y.append(label)
        if len(data_X) % 100000 == 0:
            print "Processed %d rows, loaded %d examples." % (
                cnt, len(data_X))
    cat_X = data_X
    if expand_categorical:
        encoder = OneHotEncoder(categorical_features=list(CATEGORICAL_FEATURES), sparse=False)
        cat_X = encoder.fit_transform(cat_X)
        cat_X = MaxAbsScaler().fit_transform(cat_X)
        print "Feature indices: ", encoder.feature_indices_
        print "Cat_X shape: ", cat_X.shape
    return (data_X, cat_X, np.array(data_y) if is_labeled else None)

def _mean_average_precision(predicted_probability_distribution,
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


class FastRandomForest(RandomForestClassifier):
    """This black magic is needed since sklearn random forest 
       has memory issues at prediction time. It is a simple
       bufferization over predict_proba() to avoid memory overuse."""

    BLOCK_SIZE = 10000

    def _predict_proba(self, X):
        return super(FastRandomForest, self).predict_proba(X)

    def predict_proba(self, X):
        ys = [
            self._predict_proba(X[i:i+self.BLOCK_SIZE])
            for i in xrange(0, len(X), self.BLOCK_SIZE)]
        return np.concatenate(ys)

    def score(self, X, y):
        print "score (random forest)"
        predicted_probability_distribution = self.predict_proba(X)
        return _mean_average_precision(
            predicted_probability_distribution, self.classes_, y)

class CustomKNN(KNeighborsClassifier):

    def score(self, X, y):
        print "score (knn)"
        predicted_probability_distribution = self.predict_proba(X)
        return _mean_average_precision(
            predicted_probability_distribution, self.classes_, y)

class FastLogisticRegression(LogisticRegression):

    def _predict_proba(self, X):
        return super(FastLogisticRegression, self).predict_proba(X)

    def _fit(self, X, y, sample_weight=None):
        return super(FastLogisticRegression, self).fit(X, y, sample_weight)

    def predict_proba(self, X):
        return self._predict_proba(X)

    def fit(self, X, y, sample_weight=None):
        print "fit (logistic_regression)"
        return self._fit(X, y, sample_weight)

    def score(self, X, y):
        print "score (logistic_regression)"
        predicted_probability_distribution = self.predict_proba(X)
        return _mean_average_precision(
            predicted_probability_distribution, self.classes_, y)    
