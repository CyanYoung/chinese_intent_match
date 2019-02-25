import pickle as pk

import re

import numpy as np

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from util import load_word_re, load_pair, word_replace, map_item


path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'dnn': 'model/dnn.pkl',
         'cnn_1d': 'model/cnn_1d.pkl',
         'cnn_2d': 'model/cnn_2d.pkl',
         'rnn': 'cache/rnn.pkl'}

models = {'dnn': load_model(map_item('dnn', paths)),
          'cnn_1d': load_model(map_item('cnn_1d', paths)),
          'cnn_2d': load_model(map_item('cnn_2d', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)

    core_sents = map_item(name, caches)
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    encode = map_item(name + '_encode', models)
    encode_seq = encode.predict([pad_seq])
    sims = list()
    for core_sent in core_sents:
        sims.append(1 - cos_dist(encode_seq, core_sent))
    sims = np.array(sims)
    max_sims = sorted(sims, reverse=True)[:vote]
    max_inds = np.argsort(-sims)[:vote]
    max_preds = [core_labels[ind] for ind in max_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, sim in zip(max_preds, max_sims):
            formats.append('{} {:.3f}'.format(pred, sim))
        return ', '.join(formats)
    else:
        pairs = Counter(max_preds)
        return pairs.most_common()[0][0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn', vote=5))
        print('cnn: %s' % predict(text, 'cnn', vote=5))
        print('rnn: %s' % predict(text, 'rnn', vote=5))
