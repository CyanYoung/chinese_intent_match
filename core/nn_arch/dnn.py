from keras.layers import Dense, Dropout, Lambda
from keras.layers import Flatten, Concatenate, Subtract, Multiply, Dot

import keras.backend as K


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
