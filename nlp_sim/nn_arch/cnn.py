from keras.layers import Conv1D, Conv2D, Dense
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.layers import Flatten, Reshape, Concatenate, Subtract, Multiply, Dot, Lambda

import keras.backend as K


seq_len = 30


def cnn_siam_parallel(embed_input1, embed_input2):
    ca1 = Conv1D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    da = Dense(1, activation='sigmoid')
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
    z = Dropout(0.5)(z)
    return da(z)


def cnn_siam_serial(embed_input1, embed_input2):
    ca2 = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')
    da = Dense(1, activation='sigmoid')
    mp1 = MaxPooling1D(2)
    mp2 = MaxPooling1D(2)
    flat = Flatten()
    x = ca2(embed_input1)
    x = mp1(x)
    x = ca3(x)
    x = mp2(x)
    x = flat(x)
    y = ca2(embed_input2)
    y = mp1(y)
    y = ca3(y)
    y = mp2(y)
    y = flat(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def cnn_join_parallel(embed_input1, embed_input2):
    ca1 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu')
    ca2 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
    da = Dense(1, activation='sigmoid')
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
    x = Dropout(0.5)(x)
    return da(x)


def cnn_join_serial(embed_input1, embed_input2):
    ca2 = Conv2D(filters=64, kernel_size=2, padding='same', activation='relu')
    ca3 = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
    da = Dense(1, activation='sigmoid')
    mp1 = MaxPooling2D(2)
    mp2 = MaxPooling2D(2)
    dot_input = Dot(2)([embed_input1, embed_input2])
    dot_input = Reshape((seq_len, seq_len, 1))(dot_input)
    x = ca2(dot_input)
    x = mp1(x)
    x = ca3(x)
    x = mp2(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    return da(x)
