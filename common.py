def parse(line, is_labeled=True):
    tokens = line.split(',')
    if is_labeled and int(tokens[18]) == 0:
        return None
    if is_labeled:
        tokens = tokens[:18] + tokens[20:]
    if not is_labeled:
        tokens = tokens[1:]
    features = tokens[1:11] + tokens[13:21]
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
