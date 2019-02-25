import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from util import flat_read


embed_len = 200
max_vocab = 5000
seq_len = 30

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'


def embed(sents, path_word2ind, path_word_vec, path_embed):
    model = Tokenizer(num_words=max_vocab, filters='', char_level=True)
    model.fit_on_texts(sents)
    word_inds = model.word_index
    with open(path_word2ind, 'wb') as f:
        pk.dump(model, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def align(sents):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(sents)
    return pad_sequences(seqs, maxlen=seq_len)


def vectorize(path_data, path_pair, path_label, mode):
    sent1s = flat_read(path_data, 'text1')
    sent2s = flat_read(path_data, 'text2')
    labels = flat_read(path_data, 'label')
    sents = sent1s + sent2s
    if mode == 'train':
        embed(sents, path_word2ind, path_word_vec, path_embed)
    pad_seq1s = align(sent1s)
    pad_seq2s = align(sent2s)
    pairs = (pad_seq1s, pad_seq2s)
    labels = np.array(labels)
    with open(path_pair, 'wb') as f:
        pk.dump(pairs, f)
    with open(path_label, 'wb') as f:
        pk.dump(labels, f)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_pair = 'feat/pair_train.pkl'
    path_label = 'feat/flag_train.pkl'
    vectorize(path_data, path_pair, path_label, 'train')
    path_data = 'data/test.csv'
    path_pair = 'feat/pair_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_pair, path_label, 'test')
