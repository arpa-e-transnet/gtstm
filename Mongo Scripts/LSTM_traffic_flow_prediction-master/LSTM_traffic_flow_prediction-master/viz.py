# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset):
	dataX, dataY = [], []
	for i in range(len(dataset)):
		a = dataset[i, :-1]
		dataX.append(a)
		dataY.append(dataset[i, -1])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('temp.csv',header=None)
#print dataframe.columns.values
dataframe = dataframe.loc[dataframe[39]!=-1.0]
dataset = dataframe.values
#print dataset
dataset = dataset.astype('float32')
#print dataset
#print len(dataset)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#print "*"*20
#print dataset
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print train, test
# reshape into X=t and Y=t+1
trainX_0, trainY = create_dataset(train)
testX_0, testY = create_dataset(test)
trainY_0 = trainY
#print len(trainX), len(testX)
#print testX
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX_0, (trainX_0.shape[0], 1,trainX_0.shape[1]))
testX = numpy.reshape(testX_0, (testX_0.shape[0],1, testX_0.shape[1]))



# len_train = len(trainY)
# trainY = numpy.reshape(trainY,(len_train,1))

# len_test = len(testY)
# testY = numpy.reshape(testY,(len_test,1))

# print type(trainX_0.T),trainX_0.T.shape
# print type(trainY), trainY.shape
# print type(testX_0), testX_0.shape
# print type(testY.T), testY.shape


combined_train = numpy.column_stack( (trainX_0,trainY) )
combined_test = numpy.column_stack((testX_0, testY))

# print type(combined_train), combined_train.shape
# print type(combined_test), combined_test.shape

# print scaler.inverse_transform(combined_train)
# print scaler.inverse_transform(combined_test)

trainPredict = numpy.fromfile("trainPredict.out",dtype = float)
testPredict = numpy.fromfile("testPredict.out",dtype = float)

print trainY.shape, trainPredict.shape
print testY.shape, testPredict.shape

#print trainX.shape
#print testX.shape
#print testX
# create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(1,trainX.shape[2])))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# print "Inputs : {}".format(model.input_shape)
# print "Actual inputs:{}".format(trainX.shape)
# print "Outputs:{}".format(model.output_shape)

# model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
# # # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # # invert predictions
# # trainPredict = scaler.inverse_transform(trainX_0+trainPredict)
# trainPredict = scaler.inverse_transform(numpy.column_stack( (trainX_0,trainPredict) ))[:,-1]
# trainY = scaler.inverse_transform(numpy.column_stack( (trainX_0,trainY) ))[:,-1]
# testPredict = scaler.inverse_transform(numpy.column_stack( (testX_0,testPredict) ))[:,-1]
# testY = scaler.inverse_transform(numpy.column_stack( (testX_0,testY) ))[:,-1]

# trainY = scaler.inverse_transform(trainX_0+trainY)
# testPredict = scaler.inverse_transform(testX_0+testPredict)
# testY = scaler.inverse_transform(testX_0+testY)
# # calculate root mean squared error
# print trainY.shape
# print trainPredict.shape
# print testY.shape
# print testPredict.shape

# numpy.savetxt('trainPredict.out', trainPredict, delimiter=',')
# numpy.savetxt('testPredict.out', testPredict, delimiter=',')



trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
trainPredictPlot = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
testPredictPlot = testPredict
# # plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
