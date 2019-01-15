from keras.layers import LSTM, Dense, Masking, Dropout, Activation
from keras.layers import Bidirectional, Flatten, RepeatVector, Permute
from keras.layers import Concatenate, Subtract, Multiply, Dot, Lambda

import keras.backend as K


embed_len = 200


def rnn_siam_plain(embed_input1, embed_input2):
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
    z = Dropout(0.5)(z)
    return da(z)


def rnn_siam_stack(embed_input1, embed_input2):
    mask = Masking()
    ra1 = LSTM(200, activation='tanh', return_sequences=True)
    ra2 = LSTM(200, activation='tanh')
    da = Dense(1, activation='sigmoid')
    x = mask(embed_input1)
    x = ra1(x)
    x = ra2(x)
    y = mask(embed_input2)
    y = ra1(y)
    y = ra2(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn_siam_bi(embed_input1, embed_input2):
    mask = Masking()
    ra = LSTM(200, activation='tanh')
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(1, activation='sigmoid')
    x = mask(embed_input1)
    x = ba(x)
    y = mask(embed_input2)
    y = ba(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def attend(x, y, embed_len):
    da = Dense(200, activation='tanh')
    dn = Dense(1)
    softmax = Activation('softmax')
    sum = Lambda(lambda a: K.sum(a, axis=1))
    p = da(x)
    p = dn(p)
    p = Flatten()(p)
    p = softmax(p)
    p = RepeatVector(embed_len)(p)
    p = Permute((2, 1))(p)
    x = Multiply()([x, p])
    x = sum(x)
    p = da(y)
    p = dn(p)
    p = Flatten()(p)
    p = softmax(p)
    p = RepeatVector(embed_len)(p)
    p = Permute((2, 1))(p)
    y = Multiply()([y, p])
    y = sum(y)
    return x, y


def rnn_siam_attend(embed_input1, embed_input2):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(1, activation='sigmoid')
    x = ra(embed_input1)
    y = ra(embed_input2)
    x, y = attend(x, y, embed_len)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn_siam_bi_attend(embed_input1, embed_input2):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(1, activation='sigmoid')
    x = ba(embed_input1)
    y = ba(embed_input2)
    x, y = attend(x, y, embed_len)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn_join_plain(embed_input1, embed_input2):
    ra = LSTM(30, activation='tanh')
    da = Dense(1, activation='sigmoid')
    dot_input1 = Dot(2)([embed_input1, embed_input2])
    dot_input2 = Permute((2, 1))(dot_input1)
    x1 = ra(dot_input1)
    x2 = ra(dot_input2)
    x = Concatenate()([x1, x2])
    x = Dropout(0.5)(x)
    return da(x)


def rnn_join_stack(embed_input1, embed_input2):
    ra1 = LSTM(30, activation='tanh', return_sequences=True)
    ra2 = LSTM(30, activation='tanh')
    da = Dense(1, activation='sigmoid')
    dot_input1 = Dot(2)([embed_input1, embed_input2])
    dot_input2 = Permute((2, 1))(dot_input1)
    x1 = ra1(dot_input1)
    x1 = ra2(x1)
    x2 = ra1(dot_input2)
    x2 = ra2(x2)
    x = Concatenate()([x1, x2])
    x = Dropout(0.5)(x)
    return da(x)


def rnn_join_bi(embed_input1, embed_input2):
    ra = LSTM(30, activation='tanh')
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(1, activation='sigmoid')
    dot_input1 = Dot(2)([embed_input1, embed_input2])
    dot_input2 = Permute((2, 1))(dot_input1)
    x1 = ba(dot_input1)
    x2 = ba(dot_input2)
    x = Concatenate()([x1, x2])
    x = Dropout(0.5)(x)
    return da(x)
