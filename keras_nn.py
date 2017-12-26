import csv
import numpy as np
import pdb
from DataSet import DataSet
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py
import time
import os

# import pip
# pip.main(['install', 'git+https://github.com/mgzeitouni/Kartees-Data-Processing'])
# from github_com.mgzeitouni import DataSet

def create_model(predictors, target):

	input_dim = predictors.shape[1]
	#output_dim = target.shape[1]
	output_dim = 1
	model = Sequential()

	model.add(Dense(500, activation='relu', input_shape=(input_dim,)))
	model.add(Dense(500, activation='relu'))
	# model.add(Dense(1000, activation='relu'))


	model.add(Dense(output_dim))
	model.compile(optimizer='adam',
	              loss='mse',
	              metrics=['mse'])

	early_stopping_monitor = EarlyStopping(patience=2, monitor='val_loss')

	filepath="weights.best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', save_best_only=True, mode='max')

	training = model.fit(predictors, target, verbose=True, batch_size = 10, epochs = 10, validation_split = 0.2,callbacks=[early_stopping_monitor, checkpoint])

	#print("Loss: %s, MSE: %s" %(training.history['val_loss'], training.history['mean_squared_error'] ) )
	#print (training.history['val_loss'])
	mse = training.history['mean_squared_error'][len(training.history['mean_squared_error'])-1]
	# Create the plot
	# plt.plot(training.history['val_loss'], 'r')
	# plt.xlabel('Epochs')
	# plt.ylabel('Validation score')
	# plt.show()

	return model, mse

def load_data_sets(data_dir):

	for subdir, dirs, files in os.walk(data_dir):

		for file in files:

			# Load this file's data

			path = "%s/%s" %(data_dir,file)
			data = DataSet(path,header=True)

			predictors,target = data.sliding_window_training_set()

			pdb.set_trace()
			print(predictors)
		


if __name__=='__main__':


	# load_data_sets('../sample_data')


	# path = '../sample_data/sample1.csv'

	# data1 = DataSet(path, header=True)

	# data1.process_time_data(days_back=60, extrapolate_method='connect_points')

	# predictors,target = data1.sliding_window_training_set()

	# time_series_length = predictors.shape[1]


	# Add Ones and Zeros col for game type

	#predictors1 = np.append(predictors1,np.ones([predictors1.shape[0],1]),1)
	#predictors1 = np.append(predictors1,np.zeros([predictors1.shape[0],1]),1)

	# path = '../sample_data/sample1.csv'
	
	# data2 = DataSet(path, header=True)

	# data2.process_time_data(days_back=60, extrapolate_method='connect_points')

	# predictors2,target2 = data2.training_set(output_type='zeros')

	# # Add Zeros and Ones col for game type

	# predictors2 = np.append(predictors2,np.zeros([predictors2.shape[0],1]),1)
	# predictors2 = np.append(predictors2,np.ones([predictors2.shape[0],1]),1)

	# predictors = np.concatenate((predictors1,predictors2))
	# target = np.concatenate((target1,target2))

	#pdb.set_trace()
	#print(predictors)
	start_reading = time.time()


	input_path = 'training_sets/all/inputs/1503367642.csv'
	output_path = 'training_sets/all/outputs/1503367642.csv'
	
	print("Reading inputs... ")

	with open(input_path, 'rU') as input_file:
		
		reader = csv.reader(input_file)

		predictors = np.array([row for row in reader])

	print("Reading outputs... ")

	with open(output_path, 'rU') as output_file:
		
		reader = csv.reader(output_file)

		target = np.array([row for row in reader])

	cols_ahead_prediction = 10
	
	target = target[:,cols_ahead_prediction]

	end_reading = time.time()

	print ("Reading files time: %0.2f minutes" %((end_reading-start_reading)/60))

	print("Input shape: %s" %str(predictors.shape))
	print("Output shape: %s" %str(target.shape))

	start_training = time.time()

	model, mse = create_model(predictors, target)

	end_training = time.time()

	print ("Training time: %0.2f minutes" %((end_training-start_training)/60))

	#row = int(input("Enter number between 0 and %s: " %predictors.shape[0]))

	predictions = []

	for row in range(predictors.shape[0]):

		pred_data = predictors[row,:]

		pred_data = np.reshape(pred_data, (1,predictors.shape[1]))
		#pdb.set_trace()
		#print (model.predict(pred_data))
		predictions.append(model.predict(pred_data)[0])
		
		
	#print ("Prediction: %s, Actual: %s" %(predictions,actual))
	

	plt.plot(predictions, 'r', target, 'b' )
	plt.show()


