#!/usr/bin/env python
import csv

WHITELISTED_COLUMNS = [
  'user_location_country', 
  'user_location_region', 
  'user_location_city', 
  'orig_destination_distance',
  'hotel_market',
]

GET_KEY = lambda row: ",".join([row[column] for column in WHITELISTED_COLUMNS])
NON_EMPTY_ORIG_DEST = lambda row:row['orig_destination_distance'] != ''
IS_BOOKING = lambda row:row['is_booking'] == '1'

def read_file(file_name, 
	          max_cnt, 
	          key_filter=lambda x:True, 
	          row_filter=lambda x:True):
	total = 0
	data = {}
	with open(file_name, 'rb') as xfile:
		cnt = 0
		for row in csv.DictReader(xfile):
			cnt += 1
			if cnt % 1000000 == 0:
				print "Loaded %d rows from '%s'" % (cnt, file_name)
			key = GET_KEY(row)
			if key_filter(key) and row_filter(row):
				total += 1
				if 'hotel_cluster' in row:
					s = data.setdefault(key, set())
					if len(s) < 5:
						s.add(row['hotel_cluster'])
				else:
					data[key] = None

			if cnt >= max_cnt:
				break
	return data, len(data) * 1. / total, total * 1. / cnt

def write_file(input_file_name,
			   output_file_name, 
	           lookup,
	           model_results_file_name=None):
	model_lookup = None
	if model_results_file_name is not None:
		model_lookup = {}
		with open(model_results_file_name, "r") as model_results_file:
			for line in model_results_file.readlines()[1:]:
				[idx, hotel_cluster] = line.split(",")
				model_lookup[int(idx)] = hotel_cluster
	classified = 0
	with open(input_file_name, "r") as in_file: 
		with open(output_file_name, "w") as out_file:
			out_file.write("id,hotel_cluster\n")
			cnt = 0
			for row in csv.DictReader(in_file):
				if cnt % 1000000 == 0:
					print "Written %d rows to '%s'" % (cnt, output_file_name)
				key = GET_KEY(row)
				target_classes = lookup.get(key, None)
				if target_classes != None:
					classified += 1
					target_class = " ".join(target_classes)
				else:
					if model_lookup is not None:
						target_class = model_lookup[cnt]
					else:
						target_class = "1"
				out_file.write("%d,%s\n" % (cnt, target_class))
				cnt += 1

	return classified, cnt

MAX_CNT = 100000000


test_data, test_uniqueness, test_coverage = \
	read_file(file_name='test.csv', 
			  max_cnt=MAX_CNT)
print "Test uniqueness %.2f" % (100 * test_uniqueness) 
print "Test coverage %.2f" % (100 * test_coverage) 
train_data, train_unqueness, train_coverage = \
	read_file(file_name='train.csv', 
			  max_cnt=MAX_CNT, 
			  key_filter=lambda key:key in test_data, 
			  row_filter=lambda row:True)
print "Train uniqueness %.2f" % (100 * test_uniqueness) 
print "Train coverage %.2f" % (100 * test_coverage) 


classified, total = write_file(
	input_file_name='test.csv', 
	model_results_file_name='result.csv', 
	output_file_name='final_predictions.csv', 
	lookup=train_data)
print "Classified %d out of %d (%.1f %%)" % (classified, total, classified * 100.0 / total)

# print list(train_data)[0:4]
# print list(test_data)[0:4]