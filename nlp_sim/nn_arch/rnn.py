from keras.layers import LSTM, Dense, Bidirectional
from keras.layers import BatchNormalization, Dropout, Concatenate, Flatten
from keras.layers import Permute, Subtract, Multiply, Lambda

import keras.backend as K


def rnn_siam_plain(embed_input1, embed_input2):
    ra = LSTM(300, activation='tanh')
    fc1 = Dense(100, activation='relu')
    fc2 = Dense(1, activation='sigmoid')
    x = ra(embed_input1)
    y = ra(embed_input2)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    z = fc1(z)
    z = Dropout(0.5)(z)
    return fc2(z)


def rnn_siam_stack(embed_input1, embed_input2):
    ra1 = LSTM(300, activation='tanh', return_sequences=True)
    ra2 = LSTM(300, activation='tanh')
    fc1 = Dense(100, activation='relu')
    fc2 = Dense(1, activation='sigmoid')
    x = ra1(embed_input1)
    x = BatchNormalization()(x)
    x = ra2(x)
    x = BatchNormalization()(x)
    y = ra1(embed_input2)
    y = BatchNormalization()(y)
    y = ra2(y)
    y = BatchNormalization()(y)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    z = fc1(z)
    z = Dropout(0.5)(z)
    return fc2(z)


def attention(input, seq_len, reduce):
    x = Permute((2, 1))(input)
    x = Dense(seq_len, activation='softmax')(x)
    probs = Permute((2, 1))(x)
    output = Multiply()([input, probs])
    if reduce:
        output = Lambda(lambda a: K.mean(a, axis=1))(output)
    else:
        output = Flatten()(output)
    return output


def rnn_siam_attend(embed_input1, embed_input2, seq_len=30):
    ra = LSTM(300, activation='tanh', return_sequences=True)
    fc1 = Dense(100, activation='relu')
    fc2 = Dense(1, activation='sigmoid')
    x = ra(embed_input1)
    x = attention(x, seq_len, reduce=True)
    y = ra(embed_input2)
    y = attention(y, seq_len, reduce=True)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    z = fc1(z)
    z = Dropout(0.5)(z)
    return fc2(z)


def rnn_siam_bi_attend(embed_input1, embed_input2, seq_len=30):
    ra = LSTM(300, activation='tanh', return_sequences=True)
    ba = Bidirectional(ra, merge_mode='concat')
    fc1 = Dense(100, activation='relu')
    fc2 = Dense(1, activation='sigmoid')
    x = ba(embed_input1)
    x = attention(x, seq_len, reduce=True)
    y = ba(embed_input2)
    y = attention(y, seq_len, reduce=True)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    z = fc1(z)
    z = Dropout(0.5)(z)
    return fc2(z)
