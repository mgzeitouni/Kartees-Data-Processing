from DataSet import DataSet
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import pdb

#path = 'structured_data/structured-spark-output.csv'
path = 'sample_data/sample1.csv'

with open(path, 'rU') as data_file:

	reader = csv.reader(data_file)

	header = next(reader,None)

	data = np.array([row for row in reader])


processed_data = DataSet(data, section_col=1, time_diff_col=3, values_col=2,series_type='moving_average', ma_window_width=2, new_day_interval=0.5, shrink_set=True)
print (len(processed_data.processed_time_series))

#processed_data.sliding_window_training_set(differenced=False)

long_data = DataSet(data, section_col=1, time_diff_col=3, values_col=2,series_type='regular')
print (len(long_data.processed_time_series))


fig = plt.figure(1)
fig.suptitle("Mets Home Opener 2017", fontsize="x-large")
plt.title('Mets Home Opener, 2017')
x = plt.subplot(211)
x.set_title("Avg Section Price through time - every 10 minutes")
x.set_ylabel('Citifield, Baseline Silver 107')

plt.plot(long_data.processed_time_series[:,0],long_data.processed_time_series[:,1])



y = plt.subplot(212)
y.set_title("Avg Price through time - smoothed series")
y.set_ylabel('Citifield, Baseline Silver 107')


plt.plot(processed_data.processed_time_series[:,0],processed_data.processed_time_series[:,1])



plt.xlabel('Time to Game')

#plt.ylabel('Citifield, Baseline Silver 107')

plt.show()
