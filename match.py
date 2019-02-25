import pickle as pk

import re

import numpy as np

from keras.preprocessing.sequence import pad_sequences

from util import load_word_re, load_pair, word_replace, map_item


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        core_sents = pk.load(f)
    return core_sents


seq_len = 30

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

paths = {'dnn': 'cache/dnn.pkl',
         'cnn': 'cache/cnn.pkl',
         'rnn': 'cache/rnn.pkl'}

models = {'dnn_encode': load_encode('dnn', embed_mat, seq_len),
          'cnn_encode': load_encode('cnn', embed_mat, seq_len),
          'rnn_encode': load_encode('rnn', embed_mat, seq_len)}


def predict(text, name, vote):
    text = re.sub(stop_word_re, '', text.strip())
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
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
