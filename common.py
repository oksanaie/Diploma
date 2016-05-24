import argparse

def parse(line, is_labeled=True):
    tokens = line.split(',')
    features = [tokens[3], tokens[4], tokens[5], tokens[13], tokens[14], tokens[15], tokens[16], 
        tokens[17], tokens[20], tokens[21], tokens[22]]
    features = [float (f) for f in features]
    if is_labeled == False:
    	return (features, None)
    else: 
    	label = int (tokens[23])
    return (features, label)

def load_dataset_from_file(filename, examples_count, is_labeled=True):
	data = open (filename, 'r')
	data_X = []
	data_y = []
	cnt = -1
	for line in data: 
	    cnt += 1
	    if cnt == 0:
	        continue
	    (features, label) = parse(line, is_labeled)
	    if cnt <= examples_count:
	        data_X.append(features)
	        data_y.append(label)
	    else: 
	        break
	if is_labeled == False:
		return (data_X, None)
	else:
		return (data_X, data_y)