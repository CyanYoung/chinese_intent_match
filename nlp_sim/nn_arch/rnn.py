from keras.layers import LSTM, Dense, Bidirectional, Masking
from keras.layers import Dropout, Concatenate, Flatten, Permute
from keras.layers import Subtract, Multiply, Dot, Lambda

import keras.backend as K


seq_len = 30


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


def attend(input, seq_len, reduce):
    x = Permute((2, 1))(input)
    x = Dense(seq_len, activation='softmax')(x)
    probs = Permute((2, 1))(x)
    output = Multiply()([input, probs])
    if reduce:
        output = Lambda(lambda a: K.mean(a, axis=1))(output)
    else:
        output = Flatten()(output)
    return output


def rnn_siam_attend(embed_input1, embed_input2):
    ra = LSTM(200, activation='tanh', return_sequences=True)
    da = Dense(1, activation='sigmoid')
    x = ra(embed_input1)
    x = attend(x, seq_len, reduce=True)
    y = ra(embed_input2)
    y = attend(y, seq_len, reduce=True)
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
    x = attend(x, seq_len, reduce=True)
    y = ba(embed_input2)
    y = attend(y, seq_len, reduce=True)
    diff = Lambda(lambda a: K.abs(a))(Subtract()([x, y]))
    prod = Multiply()([x, y])
    z = Concatenate()([x, y, diff, prod])
    z = Dropout(0.5)(z)
    return da(z)


def rnn_join_plain(embed_input1, embed_input2):
    ra = LSTM(30, activation='tanh')
    da = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = ra(dot_input)
    x = Dropout(0.5)(x)
    return da(x)


def rnn_join_stack(embed_input1, embed_input2):
    ra1 = LSTM(30, activation='tanh', return_sequences=True)
    ra2 = LSTM(30, activation='tanh')
    da = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = ra1(dot_input)
    x = ra2(x)
    x = Dropout(0.5)(x)
    return da(x)


def rnn_join_bi(embed_input1, embed_input2):
    ra = LSTM(30, activation='tanh')
    ba = Bidirectional(ra, merge_mode='concat')
    da = Dense(1, activation='sigmoid')
    dot_input = Dot(2)([embed_input1, embed_input2])
    x = ba(dot_input)
    x = Dropout(0.5)(x)
    return da(x)
