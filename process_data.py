import csv
import numpy as np
import pdb
from DataSet import DataSet
import os
#import keras.utils
import time

def create_training_files(data_dir):

	timestamp = str(time.time())[0:10]

	os.mkdir('training_sets/inputs/%s' %timestamp)
	os.mkdir('training_sets/outputs/%s' %timestamp)

	for subdir, dirs, files in os.walk(data_dir):

		for file in files:

			sample_name = file.replace('.csv', '')

			# Load this file's data

			if '.DS_Store' not in file:

				path = "%s/%s" %(data_dir,file)

				print(path)

				with open(path, 'rU') as data_file:

					reader = csv.reader(data_file)

					next(reader,None)

					data = np.array([row for row in reader])


				#processed_data = DataSet(path, data)
				processed_data = DataSet(path, data, series_type='moving_average', ma_window_width=2, day_interval=0.5, shrink_set=True)

				processed_data.sliding_window_training_set(differenced=False)

				processed_data.create_training_set()



				# Write to input file
				
				with open("training_sets/inputs/%s/%s_training.csv" %(timestamp,sample_name), 'w+') as training_file:

					writer = csv.writer(training_file)

					writer.writerows(processed_data.current_training_input)

				
				with open("training_sets/outputs/%s/%s_training.csv" %(timestamp,sample_name), 'w+') as training_file:

					writer = csv.writer(training_file)

					writer.writerows(processed_data.current_training_output )
				

	return timestamp


def consolidate_training_files(timestamp):

	start = time.time()

	#timestamp = str(time.time())[0:10]
	timestamp = str(timestamp)

	input_training_file_created = False

	for subdir, dirs, files in os.walk('training_sets/inputs/%s' %timestamp):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/inputs/%s/%s' %(timestamp, file), 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					write_type = 'a'

					if not input_training_file_created:

						write_type='w+'


					with open('training_sets/all/inputs/%s.csv' %timestamp, write_type) as all_data_file:

						writer = csv.writer(all_data_file)

						input_training_file_created = True

						writer.writerows(data)

	output_training_file_created = False


	for subdir, dirs, files in os.walk('training_sets/outputs/%s' %timestamp):

		for file in files:

			if '.DS_Store' not in file:

				with open('training_sets/outputs/%s/%s' %(timestamp, file), 'rU') as data_file:

					reader = csv.reader(data_file)

					data = np.array([row for row in reader])

					write_type = 'a'

					if not output_training_file_created:

						write_type='w+'

					with open('training_sets/all/outputs/%s.csv' %timestamp, write_type) as all_data_file:

						writer = csv.writer(all_data_file)

						output_training_file_created = True

						writer.writerows(data)



	end = time.time()

	print ("Consolidation Time: %0.2f minutes" %((end-start)/60))

if __name__=='__main__':


	timestamp = create_training_files('sample_data')
	
	consolidate_training_files(timestamp)
