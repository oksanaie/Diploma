import datetime

def date(dt_str):
    ans = datetime.date(*[int(i) for i in dt_str.split("-")])
    return ans

def parse(line, is_labeled=True):
    tokens = line.split(',')
    if is_labeled and int(tokens[18]) == 0:
        return None
    if is_labeled:
        tokens = tokens[:18] + tokens[20:]
    if not is_labeled:
        tokens = tokens[1:]

    check_in = date(tokens[11])
    check_out = date(tokens[12])
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

    features = tokens[1:11] + tokens[13:21] + extra_features
    features = [
        float (f) if f != '' else 0.0 
        for f in features]
    return (features, int(tokens[21]) if is_labeled else None)

def load_dataset_from_file(filename, examples_count, is_labeled=True):
    data = open (filename, 'r').readlines()
    data_X = []
    data_y = []
    cnt = 0
    for line in data[1:]: 
        cnt += 1
        if len(data_X) == examples_count:
            break
        parse_result = parse(line, is_labeled)
        if parse_result == None:
            continue
        (features, label) = parse_result
        data_X.append(features)
        data_y.append(label)
        if len(data_X) % 100000 == 0:
            print "Processed %d rows, loaded %d examples." % (
                cnt, len(data_X))
    return (data_X, data_y if is_labeled else None)
