from keras.layers import Dense, Conv1D, Conv2D, LSTM, Lambda
from keras.layers import Dropout, GlobalMaxPooling1D, GlobalMaxPooling2D, Masking, Flatten
from keras.layers import Permute, Reshape, Concatenate, Subtract, Multiply, Dot

import keras.backend as K


seq_len = 30


def dnn_siam(embed_input1, embed_input2):
    mean = Lambda(lambda a: K.mean(a, axis=1))
    da1 = Dense(200, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(1, activation='sigmoid')
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.2)(z)
    return da3(z)


def dnn_join(embed_input1, embed_input2):
    da1 = Dense(500, activation='relu')
    da2 = Dense(200, activation='relu')
    da3 = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = Flatten()(dot_input)
    x = da1(x)
    x = da2(x)
    x = Dropout(0.2)(x)
    return da3(x)


def cnn_siam(embed_input1, embed_input2):
    ca1 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    da1 = Dense(200, activation='relu')
    da2 = Dense(1, activation='sigmoid')
    mp = GlobalMaxPooling1D()
    concat1 = Concatenate()
    concat2 = Concatenate()
    x1 = ca1(embed_input1)
    x1 = mp(x1)
    x2 = ca2(embed_input1)
    x2 = mp(x2)
    x3 = ca3(embed_input1)
    x3 = mp(x3)
    x = concat1([x1, x2, x3])
    y1 = ca1(embed_input2)
    y1 = mp(y1)
    y2 = ca2(embed_input2)
    y2 = mp(y2)
    y3 = ca3(embed_input2)
    y3 = mp(y3)
    y = concat1([y1, y2, y3])
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = concat2([x, y, diff, prod])
    z = da1(z)
    z = Dropout(0.2)(z)
    return da2(z)


def cnn_join(embed_input1, embed_input2):
    ca1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
    da1 = Dense(200, activation='relu')
    da2 = Dense(1, activation='sigmoid')
    mp = GlobalMaxPooling2D()
    dot_input = Dot(2)([embed_input1, embed_input2])
    dot_input = Reshape((seq_len, seq_len, 1))(dot_input)
    x1 = ca1(dot_input)
    x1 = mp(x1)
    x2 = ca2(dot_input)
    x2 = mp(x2)
    x3 = ca3(dot_input)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    x = da1(x)
    x = Dropout(0.2)(x)
    return da2(x)


def rnn_siam(embed_input1, embed_input2):
    mask = Masking()
    ra = LSTM(200, activation='tanh')
    da = Dense(1, activation='sigmoid')
    x = mask(embed_input1)
    x = ra(x)
    y = mask(embed_input2)
    y = ra(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.2)(z)
    return da(z)


def rnn_join(embed_input1, embed_input2):
    ra = LSTM(30, activation='tanh')
    da = Dense(1, activation='sigmoid')
    dot_input1 = Dot(2)([embed_input1, embed_input2])
    dot_input2 = Permute((2, 1))(dot_input1)
    x1 = ra(dot_input1)
    x2 = ra(dot_input2)
    x = Concatenate()([x1, x2])
    x = Dropout(0.2)(x)
    return da(x)
