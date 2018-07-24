import os, math, time, argparse

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

BATCH_SIZE = 100  # batch size
HIDDEN_SIZE = 150  # the number of LSTM units
EPOCHS = 10  # the number of epoch

STEPS_PAR_CYCLE = 50  
NUMBERS_OF_CYCLE = 100

Ls = 5  # Ls steps ahead forecasting
Tau = 10  # time interval of the input window

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--plt', action='store_true', help='flag for whether to plot results')
args = parser.parse_args()

PLT = args.plt

# normalizing the sequences
def Normalize(arr):

	# vmin = min(arr)
	vmin = arr.min()
	# vmax = max(arr)
	vmax = arr.max()
	norm = (arr - vmin) / (vmax - vmin)

	return norm

# generate the simulated input sequences
def generate_sine_data():

	df = pd.DataFrame(np.arange(STEPS_PAR_CYCLE * NUMBERS_OF_CYCLE + Tau + Ls), columns=['t'])
	df['sine'] = df.t.apply(lambda t: math.sin(t*(2*math.pi/STEPS_PAR_CYCLE)))
	df['sine_rand'] = df.t.apply(lambda t: math.sin(t*(2*math.pi/STEPS_PAR_CYCLE))) + np.random.rand(len(df))
	df['sine_int'] = df.t.apply(lambda t: math.sin(t*(2*math.pi/STEPS_PAR_CYCLE))) + 2.

	return df[['sine','sine_rand','sine_int']]

# create dataset for learning DNN
def create_dataset(dataset):

	dataX, dataY = [], []
	for i in range(len(dataset)-Tau-Ls-1):
		tmp = dataset[i:i+Tau]
		dataX.append(tmp)
		dataY.append(dataset[i+Tau+Ls])

	return np.array(dataX), np.array(dataY)

def main():

	start = time.time()

	# generate multiple time-series sequences
	dataframe = generate_sine_data()

	dataset = dataframe.values.astype('float32')
	# put dataset of multiple time-series sequences
	dataset = Normalize(dataset)

	# create dataset 
	length = len(dataset)
	train_size = int(length * 0.67)
	test_size = length - train_size
	train, test = dataset[:train_size], dataset[train_size:]

	trainX, trainY = create_dataset(train)
	testX, testY = create_dataset(test)

	trainX = trainX[len(trainX)%BATCH_SIZE:]
	trainY = trainY[len(trainY)%BATCH_SIZE:]
	length_test = len(testX)
	testX = testX[len(testX)%BATCH_SIZE:]
	testY = testY[len(testY)%BATCH_SIZE:]

	trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],3))
	testX = np.reshape(testX, (testX.shape[0],testX.shape[1],3))

	# construct the DNN model (LSTM + fully_connected_layer)
	model = Sequential()
	model.add(LSTM(HIDDEN_SIZE, batch_input_shape=(BATCH_SIZE, Tau, 3)))
	model.add(Dense(3))
	model.summary()
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	# learn the DNN model on training dataset
	hist = model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0, shuffle=True)

	# plot the leargning curve
	if PLT:
		epochs = range(1,11)
		plt.figure()
		plt.plot(epochs, hist.history['loss'], label='loss/training')
		plt.plot(epochs, hist.history['acc'], label='acc/training')
		plt.xlabel('epoch'); plt.ylabel('acc / loss')
		plt.legend(); plt.show()
		plt.close()

	# evaluate the DNN model on test dataset
	score = model.evaluate(testX, testY, batch_size=BATCH_SIZE, verbose=0)
	print('loss: {0[0]}, acc: {0[1]} on test dataset'.format(score))

	# forecast Ls-steps-ahead value on the test dataset
	predicted = model.predict(testX, batch_size=BATCH_SIZE)

	# plot testY and predictedY
	if PLT:
		df_out = pd.DataFrame(predicted[:200])
		df_out.columns = ["predicted_sine",'predicted_sine_rand','predicted_sine_int']
		df_out = pd.concat([df_out,pd.DataFrame(testY[:200],columns=["input_sine","input_sine_rand","input_sine_int"])])
		plt.figure(); df_out.plot(); plt.show()
		plt.close()

	# plot the forecasting results on test dataset
	if PLT:
		plt.ion()
		i = 0
		while i < 20:
			K = 3 # the number of sensor
			fig = plt.figure(figsize=(8,K*4))
			plt.subplots_adjust(hspace=0.2)
			for j in range(K):
				plt.subplot(K,1,j+1)
				plt.plot(range(i,i+Tau+Ls+1), test[length_test%BATCH_SIZE:][i:i+Tau+Ls+1,j],color='silver', label='original')
				plt.plot(range(i,i+Tau),testX[i,:,j], color='dodgerblue', label='input')
				plt.scatter(i+Tau+Ls, predicted[i,j], s=15, color='orange',label='forecast')
				plt.legend()
			plt.draw(); plt.pause(1.2); plt.clf()
			i += 1
		plt.close()

	end = time.time()
	print('elapsed_time: {}[s]'.format(end-start))

if __name__ == '__main__':
	main()