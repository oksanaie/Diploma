import datetime
import numpy as np

from features import *
from sklearn.ensemble import RandomForestClassifier

def date(dt_str):
    try:
        return datetime.datetime.strptime(dt_str, "%Y-%m-%d")  
    except ValueError as e:
        print "Non-fatal error: %s" % e
        return datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")  

RAW_FEATURES = [
    SITE_NAME, 
    POSA_CONTINENT, 
    USER_LOCATION_COUNTRY, 
    USER_LOCATION_REGION,
    USER_LOCATION_CITY,
    ORIG_DESTINATION_DISTANCE,
    USER_ID,
    IS_MOBILE,
    IS_PACKAGE,
    CHANNEL,
    SRCH_ADULTS_CNT,
    SRCH_CHILDREN_CNT,
    SRCH_RM_CNT,
    SRCH_DESTINATION_ID,
    SRCH_DESTINATION_TYPE_ID,
    HOTEL_CONTINENT,
    HOTEL_COUNTRY,
    HOTEL_MARKET]

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

def get_features(line, is_labeled=True):
    (tokens, label) = parse_line(line, is_labeled)
    if tokens is None:
        return None
    check_in = date(tokens[SRCH_CI])
    check_out = date(tokens[SRCH_CO])
    len_of_stay = int(check_out.strftime('%j')) - int(check_in.strftime('%j'))
    if len_of_stay < 0:
       len_of_stay += 365
    weekends = 0
    if (check_in.weekday() == 4 or check_in.weekday() == 5) and (len_of_stay < 4):
        weekends = 1
    extra_features = [len_of_stay, weekends,
                  int(check_in.strftime('%j')), int(check_out.strftime('%j')), 
                  int(check_in.month), int(check_out.month),
                  int(check_in.strftime('%W')), int(check_out.strftime('%W'))]

    features = [
        tokens[raw_feature_id]
        for raw_feature_id in RAW_FEATURES] + extra_features
    features = [
        float (f) if f != '' else 0.0 
        for f in features]
    return (features, label)

def load_dataset_from_file(filename, examples_count, is_labeled=True):
    data = open (filename, 'r').readlines()
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
    return (np.array(data_X), np.array(data_y) if is_labeled else None)

class FastRandomForest(RandomForestClassifier):
    """This black magic is needed since sklearn random forest 
       has memory issues at prediction time."""

    BLOCK_SIZE = 10000

    def _predict_proba(self, X):
        return super(FastRandomForest, self).predict_proba(X)

    def predict_proba(self, X):
        print "invoking fast predict_proba"
        ys = [
            self._predict_proba(X[i:i+self.BLOCK_SIZE])
            for i in xrange(0, len(X), self.BLOCK_SIZE)]
        return np.concatenate(ys)
