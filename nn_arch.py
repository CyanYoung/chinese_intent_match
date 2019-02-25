from keras.layers import Dense, SeparableConv1D, LSTM
from keras.layers import GlobalMaxPooling1D, Masking, Lambda, Dot, Concatenate

import keras.backend as K


def dnn(embed_input1, embed_input2):
    mean = Lambda(lambda a: K.mean(a, axis=1), name='mean')
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    x = mean(embed_input1)
    x = da1(x)
    x = da2(x)
    y = mean(embed_input2)
    y = da1(y)
    y = da2(y)
    return Dot(1, normalize=True)([x, y])


def dnn_encode(embed_input):
    mean = Lambda(lambda a: K.mean(a, axis=1), name='mean')
    da1 = Dense(200, activation='relu', name='encode1')
    da2 = Dense(200, activation='relu', name='encode2')
    x = mean(embed_input)
    x = da1(x)
    return da2(x)


def cnn(embed_input1, embed_input2):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu', name='conv1')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu', name='conv2')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3')
    mp = GlobalMaxPooling1D()
    concat = Concatenate()
    da = Dense(200, activation='relu', name='encode')
    x1 = ca1(embed_input1)
    x1 = mp(x1)
    x2 = ca2(embed_input1)
    x2 = mp(x2)
    x3 = ca3(embed_input1)
    x3 = mp(x3)
    x = concat([x1, x2, x3])
    x = da(x)
    y1 = ca1(embed_input2)
    y1 = mp(y1)
    y2 = ca2(embed_input2)
    y2 = mp(y2)
    y3 = ca3(embed_input2)
    y3 = mp(y3)
    y = concat([y1, y2, y3])
    y = da(y)
    return Dot(1, normalize=True)([x, y])


def cnn_encode(embed_input):
    ca1 = SeparableConv1D(filters=64, kernel_size=1, padding='same', activation='relu', name='conv1')
    ca2 = SeparableConv1D(filters=64, kernel_size=2, padding='same', activation='relu', name='conv2')
    ca3 = SeparableConv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3')
    mp = GlobalMaxPooling1D()
    da = Dense(200, activation='relu', name='encode')
    x1 = ca1(embed_input)
    x1 = mp(x1)
    x2 = ca2(embed_input)
    x2 = mp(x2)
    x3 = ca3(embed_input)
    x3 = mp(x3)
    x = Concatenate()([x1, x2, x3])
    return da(x)


def rnn(embed_input1, embed_input2):
    mask = Masking()
    ra = LSTM(200, activation='tanh', name='encode1')
    da = Dense(200, activation='relu', name='encode2')
    x = mask(embed_input1)
    x = ra(x)
    x = da(x)
    y = mask(embed_input2)
    y = ra(y)
    y = da(y)
    return Dot(1, normalize=True)([x, y])


def rnn_encode(embed_input):
    ra = LSTM(200, activation='tanh', name='encode1')
    da = Dense(200, activation='relu', name='encode2')
    x = Masking()(embed_input)
    x = ra(x)
    return da(x)
