import csv
import numpy as np
from .algebra_functions import *
import pdb
#import keras.utils
from pandas import Series
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt


DEFAULT_HEADER = ['Time', 'Time_Diff', 'Zone_Section_Id', 'Zone_Name', 'Total_Tickets', 'Average_Price', 'Zone_Section_Total_Tickets', 'Zone_Section_Average_Price', 'Zone_Section_Min_Price', 'Zone_Section_Max_Price', 'Zone_Section_Std', 'Wins', 'Losses', 'L_10', 'Section_Median', 'Total_Listings', 'Zone_Section_Num_Listings', 'Data_Type', 'Event_Id']

def map_section(section):

	section_map = {182157:1,
			181979:0,
			181981:2}

	return section_map[section]


class DataSet:

	def __init__(self, data_matrix, section_col, time_diff_col,values_col, series_type='moving_average', ma_window_width=2, shrink_set=True, new_day_interval=0.5 ):


		self.time_processed = False

		self.time_training_set = False

		self.processed_matrix = None

		self.training_inputs = None

		self.training_outputs = None

		self.shrunk = False

		self.section = data_matrix[:,section_col][0]

		time_diff_array = np.array(data_matrix[:,time_diff_col])

		values_array = np.array(data_matrix[:,values_col])

		self.raw_time_data_matrix = np.column_stack([time_diff_array,values_array] )
		
		self.process_time_data()

		if series_type=='moving_average':

			self.ma_window_width = ma_window_width
			self.new_day_interval = new_day_interval
			self.shrink_set = shrink_set
			self.moving_average()


	def process_time_data(self, days_back=90, extrapolate_method='connect_points'):

		length = days_back * 6 * 24

		# Create array of times
		time_array = np.arange(length)

		raw_length = int(self.raw_time_data_matrix.shape[0])

		processed_length = int(time_array.shape[0])

		self.new_unprocessed_matrix = np.zeros([processed_length,2])

		self.new_unprocessed_matrix[:,0] = time_array


		# Transform raw times
		for row in self.raw_time_data_matrix:

			time = row[0]

			y = float(row[1])

			new_time = int(round(float(time)*24.0*6.0))
			
			index = 0

			# Find and replace in the new matrix
			for row in self.new_unprocessed_matrix:

				if new_time==row[0]:
					
					self.new_unprocessed_matrix[index,1] = y

				index+=1

		if extrapolate_method=='connect_points':

			self.processed_time_series = connect_points(self.new_unprocessed_matrix)
			self.processed = True
			self.time_processed =True

		last_point = int(round(days_back * 24 * 6))

		self.processed_time_series = self.processed_time_series[0:last_point, :]

		if self.raw_time_data_matrix[0,0] > self.raw_time_data_matrix[-1,0]:

			price_list = self.processed_time_series[:,1].tolist()

			price_list_reversed = list(reversed(price_list))

			self.processed_time_series[:,1] = price_list_reversed


	def full_series_training_set(self, output_type='regular'):

		# If time series not yet processed - then process

		if self.time_processed == False:
			self.process_time_data()


		# Create Input Matrix for Training Set

		n_data_points = self.processed_time_series.shape[0]

		time_array = self.processed_time_series[:,0]
		y_array = self.processed_time_series[:,1]

		self.full_series_training_outputs = np.array([y_array for i in range(len(y_array))])

		input_matrix = np.zeros([n_data_points,n_data_points])

		output_matrix = np.zeros([n_data_points,n_data_points])

		temp_input_array = np.zeros(n_data_points)

		temp_output_array = y_array


		for x in range(n_data_points):

			current_y = y_array[x]

			temp_input_array[x] = y_array[x]

			temp_output_array[x] = 0

			input_matrix[x] = temp_input_array

			output_matrix[x] = temp_output_array


		self.full_series_training_inputs = input_matrix

		if output_type=="zeros":

			self.full_series_training_inputs = output_matrix

		self.time_training_set = True

		self.current_training_input = self.full_series_training_inputs
		self.current_training_output = self.full_series_training_outputs

		return self.full_series_training_inputs,self.full_series_training_outputs 


	def sliding_window_training_set(self, differenced, base_difference, days_back, days_ahead):

		if not self.time_processed:

			self.process_time_data()


		self.len_back_sliding_window = int(days_back/self.new_day_interval)
		self.len_ahead_sliding_window = int(days_ahead/self.new_day_interval)

		if not self.shrunk:

			self.len_back_sliding_window = int(self.len_back_sliding_window*6*24)
			self.len_ahead_sliding_window = int(self.len_ahead_sliding_window*6*24)


		time_array = self.processed_time_series[:,0]
		y_array = self.processed_time_series[:,1]

		# Create matrix for training set - length of time series, and width of sliding window
		input_matrix = np.zeros([time_array.shape[0],self.len_back_sliding_window+2])

		# Input first column as current time
		input_matrix[:,0] = np.reshape(time_array, (1,time_array.shape[0]))

		# Input second column current price
		input_matrix[:,1] = np.reshape(y_array, (1,y_array.shape[0]))

		# Create output matrix - sequence for future prices
		output_matrix = np.zeros([time_array.shape[0],self.len_ahead_sliding_window])

		current_row = 0

		for x in self.processed_time_series:

			current_time = x[0]
			current_y = x[1]
			initial_last_y = current_y
			last_y = None

			# Loop through each days back and find difference in y
			for k in range(self.len_back_sliding_window):

				try:
					y_to_compare = y_array[current_row+k+1]
				except:
					y_to_compare = current_y

				if not differenced:
					comparison = 0

				elif base_difference == 'current_y':
					comparison = current_y

				else:
					if last_y == None:
						last_y = initial_last_y
					comparison = last_y

				# Find difference and input in matrix (starting in third column)
				input_matrix[current_row][k+2] = y_to_compare - comparison

				last_y = y_to_compare

			last_y = None

			# Loop ahead for outputs to find prices in future
			for k in range(self.len_ahead_sliding_window):

				# Check if we have a time ahead
				if current_row-k-1 < 0:
					
					y_to_compare = current_y

				else:
					y_to_compare = y_array[current_row-k-1]

				if not differenced:
					comparison =0

				elif base_difference == 'current_y':
					comparison = current_y

				else:
					if last_y == None:
						last_y = initial_last_y
					comparison = last_y
					

				# Find difference and put into matrix (starting in third column)
				output_matrix[current_row][k] =   y_to_compare - comparison
				last_y = y_to_compare


			current_row+=1

		self.sliding_window_training_inputs = input_matrix
		self.sliding_window_training_outputs = output_matrix

		self.time_training_set = True

		self.current_training_input = self.sliding_window_training_inputs
		self.current_training_output = self.sliding_window_training_outputs
		
		return self.sliding_window_training_inputs,self.sliding_window_training_outputs 


	def create_training_set(self, time_type='sliding_window', differenced=False, base_difference='current_y', days_back=30, days_ahead=30):

		# Create time processed training sets if not done already

		if not self.time_training_set:

			if time_type=='sliding_window':

				self.sliding_window_training_set(differenced,base_difference,days_back,days_ahead)

			else:

				self.full_series_training_set()

		# Pull out section from raw data matrix

		#section = int(self.raw_time_data_matrix[0,2])

		#section = [map_section(int(self.section))]
		section=int(self.section)
		section_col = np.full((self.current_training_input.shape[0],1),section)
		self.current_training_input = np.hstack((section_col,self.current_training_input) )

		#one_hot_labels = keras.utils.to_categorical(section,num_classes=3)

		#one_hot_labels = one_hot_labels[0] * np.ones((self.current_training_input.shape[0],1))

		#print(one_hot_labels)

		#self.current_training_input = np.hstack((one_hot_labels,self.current_training_input))

		

	def moving_average(self):

		ma_window_width = int(self.ma_window_width * 144)

		if self.time_processed == False:
			self.process_time_data()

		new_list = self.processed_time_series[:,1].tolist()

		#new_list = list(reversed(new_list))

		series = Series(new_list)
		
		rolling = series.rolling(window=ma_window_width)

		rolling_mean = rolling.mean()

		for i in rolling_mean:
			if str(i) !='nan':
				first_x = i
				break


		rolling_mean = rolling_mean.fillna(first_x)

		self.moving_average_series = rolling_mean.values

		new_series = []

		# Shrink data set

		if self.shrink_set:

			n_new_points = int((self.moving_average_series.shape[0]/144) / self.new_day_interval)

			k=0
			step = int(self.moving_average_series.shape[0]/n_new_points)
			for i in range(0,self.moving_average_series.shape[0],step):
				
				new_series.append([k,self.moving_average_series[i]])
				k+=1

			new_series = np.array(new_series)

			self.processed_time_series = new_series
			self.shrunk = True

		else:

			self.processed_time_series = self.moving_average_series






