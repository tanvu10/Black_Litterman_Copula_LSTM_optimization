import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import RNN, SimpleRNN
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers.core import Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def scaler(df):
  return((df - np.mean(df,axis=0))/(np.max(df, axis=0)-np.min(df,axis=0)))

def DL_train(df, time_step=30):
    from tensorflow.keras.layers import LSTM
    y = df['return']
    X = df.drop(columns=['return'])

    X = scaler(X)
    X.dropna(axis=1,inplace=True)
    num_factor = X.shape[1]

    today_value_X = np.array(X.iloc[-time_step:, :]).reshape(1, time_step, num_factor)

    # y, X to train
    y = np.array(y)[1:].reshape(-1, 1)
    X = X.iloc[:-1, :]

    X_new = np.zeros((X.shape[0] - time_step, time_step, num_factor))
    y_new = np.zeros((y.shape[0] - time_step,))
    for ix in range(X_new.shape[0]):
        for jx in range(time_step):
            X_new[ix, jx, :] = X.values[ix + jx, :]
        y_new[ix] = y[ix + time_step]

    print(time_step, num_factor)
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(time_step, num_factor)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])

    callback = EarlyStopping(monitor='loss', patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    LSTM_model = model.fit(X_new, y_new, batch_size=64, epochs=100, verbose=1, validation_split=0.2, callbacks=[callback, reduce_lr])

    # plt.plot(LSTM_model.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    print(model.summary())
    tomorrow_pred = model.predict(today_value_X)
    # reshaping output
    train_prediction_true = np.squeeze(tomorrow_pred)
    return train_prediction_true


if __name__ == '__main__':
    # test
    df = pd.read_csv('C:/Users/Tan Vu/Desktop/thesis/Copula_BL/data_v2/stock_data_DL_folder/CII.csv', index_col=0)
    print(df)
    prediction = DL_train(df)
    print(prediction)
