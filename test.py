from DataSet import DataSet
import numpy as np
import matplotlib.pyplot as plt
import csv

path = 'sample_data/sample1.csv'

with open(path, 'rU') as data_file:

	reader = csv.reader(data_file)

	next(reader,None)

	data = np.array([row for row in reader])


processed_data = DataSet(path, data, series_type='moving_average', ma_window_width=2, day_interval=0.5, shrink_set=True)

#processed_data.sliding_window_training_set(differenced=False)

long_data = DataSet(path, data, series_type='regular')


plt.figure(1)
plt.subplot(211)
plt.plot(processed_data.processed_time_series[:,0],processed_data.processed_time_series[:,1])

plt.figure(1)
plt.subplot(212)
plt.plot(long_data.processed_time_series[:,0],long_data.processed_time_series[:,1])
plt.show()
