#LSTM
import pandas
import numpy  as np
import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from keras import layers
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras.layers import Bidirectional



from keras import backend as K
K.tensorflow_backend._get_available_gpus()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


dataset = pandas.read_csv('seasonal15t19dataset.csv',
engine= 'python' , skipfooter=3)
dataset = dataset.values
dataset = dataset[2:,:]
season = np.zeros((len(dataset),1))
season = season.astype('str')



labledWeekDays = np.zeros((len(dataset),1))

labledHours = np.zeros((len(dataset),1))
targetData = np.zeros((len(dataset),1))
labledSeasons = np.zeros((len(dataset),1))




def preparePrice(regionCount,regionColumn=[]):
      lprice = np.zeros((len(dataset),regionCount))
      count = 0
      for regCol in regionColumn:
        for i in range(len(dataset)):
          floatData =  float(dataset[i,regCol].replace(",","."))
          lprice[i,count] = floatData;
          lprice = lprice.astype( 'float32' )
        count = count +1 
      return lprice 

encoder = LabelEncoder()
encoder.fit(["Monday", "Tuesday", "Wednesday", "Thursday","Friday","Saturday","Sunday"])
list(encoder.classes_)
labledWeekDays[:,0] = encoder.transform(dataset[:,1])
encoder.fit(dataset[:,3])
list(encoder.classes_)
labledHours[:,0] = encoder.transform(dataset[:,3])
encoder.fit(dataset[:,0])
list(encoder.classes_)
labledSeasons[:,0] = encoder.transform(dataset[:,0])
price = preparePrice(1,[5])


#inputData = np.concatenate((price,labledHours,labledSeasons),1)
inputData = np.concatenate((price,labledHours),1)
#inputData = price
targetData[:,0] = price[:,0]
print(inputData.shape, targetData.shape)
inputData = inputData.astype('float32')
targetData = targetData.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaledInputData = scaler.fit_transform(inputData)
scaledTarget = scaler.fit_transform(targetData)
look_back = 24
look_foreward = 24
n_features = 2
reframedInputData = series_to_supervised(scaledInputData, look_back, look_foreward)
reframedTarget = series_to_supervised(scaledTarget, look_back, look_foreward)
reframedInputData = reframedInputData.values
reframedTarget = reframedTarget.values
n_train_hours = int(4*360 * 24)
#train = reframedInputData[:n_train_hours, :]
#test = reframedInputData[n_train_hours:, :]
train_X = reframedInputData[:n_train_hours, 0:n_features*look_back]
test_X = reframedInputData[n_train_hours:, 0:n_features*look_back]
train_y = reframedTarget[:n_train_hours, look_back:]
test_y = reframedTarget[n_train_hours:, look_back:]
train_X = train_X.reshape((train_X.shape[0], look_back, n_features))
test_X = test_X.reshape((test_X.shape[0], look_back, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = Sequential()
#model.add(LSTM(50, activation='relu'), input_shape=(train_X.shape[1], train_X.shape[2]))
model.add(LSTM(80,activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(look_foreward))
model.compile(loss='mse', optimizer='adam')
# fit network
#for i in range(5):
#model.fit(train_X, train_y, epochs=10, batch_size=80, verbose=1, shuffle=False)
#		model.reset_states()
history = model.fit(train_X, train_y, epochs=50, batch_size=24, validation_data=(test_X, test_y), verbose=1, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

train_Predict = model.predict(train_X)
test_Predict = model.predict(test_X)
# invert predictions
train_Predict = scaler.inverse_transform(train_Predict)
train_y = scaler.inverse_transform(train_y)


test_Predict = scaler.inverse_transform(test_Predict)
test_y = scaler.inverse_transform(test_y)
print(train_y - train_Predict)
print(train_y.shape, test_y.shape, train_Predict.shape, test_Predict.shape)
print(test_y- test_Predict)


# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(train_y, train_Predict))
#print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_y, test_Predict))
print( 'Test Score: %.2f RMSE' % (testScore))
#trainScore = mean_absolute_percentage_error(train_y, trainPredict)
#print('Train Score: %.2f MAPE' % (trainScore))

testScore = mean_absolute_percentage_error(test_y, test_Predict)
print('Test Score: %.2f MAPE' % (testScore))
pltTest_y = test_y[-1:,:]
pltTestPredict = test_Predict[-1:,:]
testScore = mean_absolute_percentage_error(pltTest_y, pltTestPredict)
print('Test Score in last 24 hours: %.2f MAPE' % (testScore))
pyplot.plot(pltTest_y[0,:])
pyplot.plot(pltTestPredict[0,:])
pyplot.legend()
pyplot.show()