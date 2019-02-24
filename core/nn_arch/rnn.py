from keras.layers import LSTM, Dense, Masking, Dropout, Lambda
from keras.layers import Permute, Concatenate, Subtract, Multiply, Dot

import keras.backend as K


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
