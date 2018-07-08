import pickle as pk

import numpy as np

from keras.layers import Input, Embedding
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

from nlp_sim.util.load import load_label
from nlp_sim.util.trial import trial
from nlp_sim.util.log import get_loggers, log_state

from nlp_sim.nn_arch.dnn import dnn_siam_average
from nlp_sim.nn_arch.cnn import cnn_siam_parallel, cnn_siam_serial
from nlp_sim.nn_arch.rnn import rnn_siam_plain, rnn_siam_stack
from nlp_sim.nn_arch.rnn import rnn_siam_attend, rnn_siam_bi_attend

from nlp_sim.nn_arch.dnn import dnn_join_average, dnn_join_flat
from nlp_sim.nn_arch.cnn import cnn_join_parallel, cnn_join_serial


loggers = {'dnn': get_loggers('dnn', 'nlp_sim/info/dnn/'),
           'cnn': get_loggers('cnn', 'nlp_sim/info/cnn/'),
           'rnn': get_loggers('rnn', 'nlp_sim/info/rnn/')}

funcs = {'dnn_siam_average': dnn_siam_average,
         'cnn_siam_parallel': cnn_siam_parallel,
         'cnn_siam_serial': cnn_siam_serial,
         'rnn_siam_plain': rnn_siam_plain,
         'rnn_siam_stack': rnn_siam_stack,
         'rnn_siam_attend': rnn_siam_attend,
         'rnn_siam_bi_attend': rnn_siam_bi_attend,
         'dnn_join_average': dnn_join_average,
         'dnn_join_flat': dnn_join_flat,
         'cnn_join_parallel': cnn_join_parallel,
         'cnn_join_serial': cnn_join_serial}


def split(pad):
    with open(pad, 'rb') as f:
        pad_mat = pk.load(f)
    mat_shape = pad_mat.shape
    pad_mat1 = np.zeros((int(mat_shape[0] / 2), mat_shape[1]))
    pad_mat2 = np.zeros((int(mat_shape[0] / 2), mat_shape[1]))
    for i in range(len(pad_mat) - 1):
        if not i % 2:
            pad_mat1[int(i / 2)] = pad_mat[i]
        else:
            pad_mat2[int(i / 2)] = pad_mat[i]
    return pad_mat1, pad_mat2


def build(embed_mat, seq_len, name):
    vocab_num, embed_dim = embed_mat.shape
    embed_layer = Embedding(input_dim=vocab_num,
                            output_dim=embed_dim,
                            weights=[embed_mat],
                            input_length=seq_len,
                            trainable=False)
    input1 = Input(shape=(seq_len,), dtype='int32')
    input2 = Input(shape=(seq_len,), dtype='int32')
    embed_input1 = embed_layer(input1)
    embed_input2 = embed_layer(input2)
    if name in funcs.keys():
        output = funcs[name](embed_input1, embed_input2)
    else:
        raise KeyError
    model = Model([input1, input2], output)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def check(sent, model, pad_mat1, pad_mat2, labels, logger, epoch, name, mode):
    probs = model.predict([pad_mat1, pad_mat2], batch_size=128)
    probs = np.reshape(probs, (1, -1))[0]
    trial(sent, probs, labels, logger, '_'.join([name, str(epoch)]), mode)


def nn(paths, name, arch, epoch, mode, thre):
    logger = loggers[name]
    name = '_'.join([name, arch])
    if mode == 'train':
        with open(paths['embed'], 'rb') as f:
            embed_mat = pk.load(f)
        pad_train1, pad_train2 = split(paths['pad_train'])
        train_labels = load_label(paths['train_label'])
        pad_dev1, pad_dev2 = split(paths['pad_dev'])
        dev_labels = load_label(paths['dev_label'])
        seq_len = pad_train1.shape[1]
        model = build(embed_mat, seq_len, name)
        check_point = ModelCheckpoint(paths[name], monitor='val_loss', verbose=True, save_best_only=True)
        log_state(logger[0], name, mode)
        model.fit([pad_train1, pad_train2], train_labels,
                  batch_size=128, epochs=epoch, verbose=True, callbacks=[check_point],
                  validation_data=([pad_dev1, pad_dev2], dev_labels))
        log_state(logger[0], name, mode)
        check(paths['train_cut'], model, pad_train1, pad_train2, train_labels, logger, epoch, name, 'train')
    elif mode == 'dev':
        pad_dev1, pad_dev2 = split(paths['pad_dev'])
        dev_labels = load_label(paths['dev_label'])
        model = load_model(paths[name])
        check(paths['dev_cut'], model, pad_dev1, pad_dev2, dev_labels, logger, epoch, name, 'dev')
    elif mode == 'test':
        pad_mat1, pad_mat2 = split(paths['pad'])
        model = load_model(paths[name])
        probs = model.predict([pad_mat1, pad_mat2], batch_size=128)
        probs = np.reshape(probs, (1, -1))[0]
        return probs > thre
    else:
        raise KeyError
