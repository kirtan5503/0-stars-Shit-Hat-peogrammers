import pandas as pd
import pandas_datareader as pdr
from keras.models import Sequential
from keras.layers import Dense,LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.style.use('fivethirtyeight')
import numpy as np
import yfinance as yf
import math
from datetime import date





start="2012-01-01"
today=date.today().strftime("%Y-%m-%d")
selected_stocks=input( )


def load_data(ticker):
    data=yf.download(ticker,start,today)
    data.reset_index(inplace=True)
    return data

data=load_data(selected_stocks)
data=data.set_index('Date')
plt.figure(figsize=(16,8))
plt.title('close price ')
plt.plot(data['Close'])
plt.xlabel('date',fontsize=18)
plt.ylabel('price',fontsize=18)
#plt.show()

datar=data.filter(['Close'])
dataset=datar.values
training_data_len=math.ceil(len(dataset)*80)

#print(training_data_len)

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
#print(scaled_data)

train_data=scaled_data[0:training_data_len, :]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)
test_data=scaled_data[training_data_len - 60:, :]
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#convert back to original scale
rmse=np.sqrt(np.mean(((predictions- y_test)**2))) #calculate rmse
print("The root mean squared error is {}".format(rmse))

train=data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions

plt.figure(figsize=(16,8))
plt.tittle('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price ', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()
