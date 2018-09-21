import pickle as pk

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from nlp_sim.util.load import load_label
from nlp_sim.util.map import map_logger, map_func
from nlp_sim.util.trial import trial
from nlp_sim.util.log import log_state


batch_size = 32


def split(path_pad):
    with open(path_pad, 'rb') as f:
        pad_seqs = pk.load(f)
    mat_shape = pad_seqs.shape
    pad_seq1s = np.zeros((int(mat_shape[0] / 2), mat_shape[1]))
    pad_seq2s = np.zeros((int(mat_shape[0] / 2), mat_shape[1]))
    for i in range(len(pad_seqs) - 1):
        if not i % 2:
            pad_seq1s[int(i / 2)] = pad_seqs[i]
        else:
            pad_seq2s[int(i / 2)] = pad_seqs[i]
    return pad_seq1s, pad_seq2s


def build(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input1 = Input(shape=(seq_len,), dtype='int32')
    input2 = Input(shape=(seq_len,), dtype='int32')
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    func = map_func(name)
    output = func(embed_input1, embed_input2)
    model = Model([input1, input2], output)
    model.summary()
    plot_model(model)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def check(sent, model, pad_seq1s, pad_seq2s, labels, logger, epoch, name, mode):
    probs = model.predict([pad_seq1s, pad_seq2s], batch_size=batch_size)
    probs = np.reshape(probs, (1, -1))[0]
    trial(sent, probs, labels, logger, '_'.join([name, str(epoch)]), mode)


def nn(paths, name, arch, epoch, mode, thre=None):
    logger = map_logger(name)
    name = '_'.join([name, arch])
    if mode == 'train':
        with open(paths['embed'], 'rb') as f:
            embed_mat = pk.load(f)
        pad_train1, pad_train2 = split(paths['pad_train'])
        train_labels = load_label(paths['train_label'])
        pad_dev1, pad_dev2 = split(paths['pad_dev'])
        dev_labels = load_label(paths['dev_label'])
        seq_len = pad_train1.shape[1]
        model = build(name, embed_mat, seq_len)
        check_point = ModelCheckpoint(paths[name], monitor='val_loss', verbose=True, save_best_only=True)
        log_state(logger[0], name, mode)
        model.fit([pad_train1, pad_train2], train_labels,
                  batch_size=batch_size, epochs=epoch, verbose=True, callbacks=[check_point],
                  validation_data=([pad_dev1, pad_dev2], dev_labels))
        log_state(logger[0], name, mode)
        check(paths['train_clean'], model, pad_train1, pad_train2, train_labels, logger, epoch, name, 'train')
    elif mode == 'dev':
        pad_dev1, pad_dev2 = split(paths['pad_dev'])
        dev_labels = load_label(paths['dev_label'])
        model = load_model(paths[name])
        check(paths['dev_clean'], model, pad_dev1, pad_dev2, dev_labels, logger, epoch, name, 'dev')
    else:
        pad_seq1s, pad_seq2s = split(paths['pad'])
        model = load_model(paths[name])
        probs = model.predict([pad_seq1s, pad_seq2s])
        probs = np.reshape(probs, (1, -1))[0]
        return probs > thre
