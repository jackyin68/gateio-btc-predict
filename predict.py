from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


candle_stick_list = pd.read_csv("btc_usdt.csv")
plt.figure(figsize=(16, 8))
plt.plot(candle_stick_list['close'])
plt.ylabel('Price')
plt.title('BTC hourly price prediction based on gateio')
plt.savefig('bts_price.png')
plt.show()

time_stamp = 20
candle_stick_list = candle_stick_list[["volume", "high", "low", "open", "close"]]
train_len = int(len(candle_stick_list) * 2 / 3)
train = candle_stick_list[:train_len + time_stamp]
valid = candle_stick_list[train_len - time_stamp:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train)
x_train, y_train = [], []
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i, 4:])
x_train, y_train = np.array(x_train), np.array(y_train)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i, 4:])
x_valid, y_valid = np.array(x_valid), np.array(y_valid)


epochs = 100
batch_size = 30
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

closing_price = model.predict(x_valid)
scaler.fit_transform(pd.DataFrame(valid['close'].values))
closing_price = scaler.inverse_transform(closing_price)
y_valid = scaler.inverse_transform(y_valid)
rms = np.sqrt(np.mean(np.power((y_valid - closing_price), 2)))
print(rms)

plt.figure(figsize=(16, 8))
dict_data = {
    'close': y_valid[..., 0],
    'prediction': closing_price[..., 0]
}
data_pd = pd.DataFrame(dict_data)
plt.plot(data_pd[['close']], label='Close price')
plt.plot(data_pd[['prediction']], label='Predict close price')
plt.xlabel('Duration (Hour)')
plt.ylabel('Bitcoin Price (USDT)')
plt.title('BTC hourly price prediction based on Gateio')
plt.legend()
plt.savefig('bts_price_predict.png')
plt.show()
