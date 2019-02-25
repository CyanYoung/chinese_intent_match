from keras.layers import Dense, Conv1D, Conv2D, LSTM, Lambda
from keras.layers import Dropout, GlobalMaxPooling1D, MaxPooling2D, Masking, Concatenate
from keras.layers import Flatten, Reshape, Subtract, Multiply, Dot

import keras.backend as K


seq_len = 30


def dnn(embed_input1, embed_input2):
    mean = Lambda(lambda a: K.mean(a, axis=1))
    da1 = Dense(200, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(200, activation='relu')
    da4 = Dense(1, activation='sigmoid')
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = da3(z)
    z = Dropout(0.2)(z)
    return da4(z)


def cnn_1d(embed_input1, embed_input2):
    ca1 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    mp = GlobalMaxPooling1D()
    da1 = Dense(200, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(1, activation='sigmoid')
    x1 = ca1(embed_input1)
    x1 = mp(x1)
    x2 = ca2(embed_input1)
    x2 = mp(x2)
    x3 = ca3(embed_input1)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    x = da1(x)
    y1 = ca1(embed_input2)
    y1 = mp(y1)
    y2 = ca2(embed_input2)
    y2 = mp(y2)
    y3 = ca3(embed_input2)
    y3 = mp(y3)
    y = Concatenate()([y1, y2, y3])
    y = da1(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = da2(z)
    z = Dropout(0.2)(z)
    return da3(z)


def cnn_2d(embed_input1, embed_input2):
    ca1 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca2 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
    mp1 = MaxPooling2D(3)
    mp2 = MaxPooling2D(5)
    da1 = Dense(200, activation='relu')
    da2 = Dense(1, activation='sigmoid')
    x = Dot(2)([embed_input1, embed_input2])
    x = Reshape((seq_len, seq_len, 1))(x)
    x = ca1(x)
    x = mp1(x)
    x = ca2(x)
    x = mp2(x)
    x = Flatten()(x)
    x = da1(x)
    x = Dropout(0.2)(x)
    return da2(x)


def rnn(embed_input1, embed_input2):
    ra = LSTM(200, activation='tanh')
    da1 = Dense(200, activation='relu')
    da2 = Dense(1, activation='sigmoid')
    x = Masking()(embed_input1)
    x = ra(x)
    y = Masking()(embed_input2)
    y = ra(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = da1(z)
    z = Dropout(0.2)(z)
    return da2(z)
