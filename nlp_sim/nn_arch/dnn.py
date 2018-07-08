from keras.layers import Dense, Dropout
from keras.layers import Concatenate, Subtract, Multiply, Dot, Lambda, Flatten

import keras.backend as K


def dnn_siam_average(embed_input1, embed_input2):
    fc1 = Dense(300, activation='relu')
    fc2 = Dense(300, activation='relu')
    fc3 = Dense(100, activation='relu')
    fc4 = Dense(1, activation='sigmoid')
    x = Lambda(lambda a: K.mean(a, axis=1))(embed_input1)
    x = fc1(x)
    x = fc2(x)
    y = Lambda(lambda a: K.mean(a, axis=1))(embed_input2)
    y = fc1(y)
    y = fc2(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    z = fc3(z)
    z = Dropout(0.5)(z)
    return fc4(z)


def dnn_join_average(embed_input1, embed_input2):
    fc1 = Dense(50, activation='relu')
    fc2 = Dense(50, activation='relu')
    fc3 = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = Lambda(lambda a: K.mean(a, axis=1))(dot_input)
    x = fc1(x)
    x = Dropout(0.5)(x)
    x = fc2(x)
    x = Dropout(0.5)(x)
    return fc3(x)


def dnn_join_flat(embed_input1, embed_input2):
    fc1 = Dense(1000, activation='relu')
    fc2 = Dense(1000, activation='relu')
    fc3 = Dense(100, activation='relu')
    fc4 = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = Flatten()(dot_input)
    x = fc1(x)
    x = Dropout(0.5)(x)
    x = fc2(x)
    x = Dropout(0.5)(x)
    x = fc3(x)
    x = Dropout(0.5)(x)
    return fc4(x)
