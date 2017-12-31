from DataSet import DataSet
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import pdb
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


path = 'sample_data/sample1.csv'

with open(path, 'rU') as data_file:

	reader = csv.reader(data_file)

	header = next(reader,None)

	data = np.array([row for row in reader])


processed_data = DataSet(data, header, series_type='moving_average', ma_window_width=2, new_day_interval=0.5, shrink_set=True)
short_set = processed_data.processed_time_series[:,0]

#processed_data.sliding_window_training_set(differenced=False)

long_data = DataSet(data, series_type='regular')
long_set = long_data.processed_time_series[:,0]

print(long_set)

# # Split the data into training/testing sets
# diabetes_X_train = diabetes_X[:-20]
# diabetes_X_test = diabetes_X[-20:]

# # Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

# # Create linear regression object
# regr = linear_model.LinearRegression()