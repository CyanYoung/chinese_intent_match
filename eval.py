import pickle as pk

import numpy as np

from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.layers import Input, Embedding

from nn_arch import dnn, cnn_1d, cnn_2d, rnn

from match import predict

from util import flat_read, map_item


def define_model(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input1 = Input(shape=(seq_len,))
    input2 = Input(shape=(seq_len,))
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2)
    return Model([input1, input2], output)


def load_model(name, embed_mat, seq_len):
    model = define_model(name, embed_mat, seq_len)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


seq_len = 30

path_test = 'data/test.csv'
path_label = 'feat/label_test.pkl'
path_embed = 'feat/embed.pkl'
texts = flat_read(path_test, 'text')
with open(path_label, 'rb') as f:
    labels = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

path_test_pair = 'data/test_pair.csv'
path_pair = 'feat/pair_test.pkl'
path_flag = 'feat/flag_test.pkl'
text1s = flat_read(path_test_pair, 'text1')
text2s = flat_read(path_test_pair, 'text2')
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_flag, 'rb') as f:
    flags = pk.load(f)

funcs = {'dnn': dnn,
         'cnn': cnn,
         'rnn': rnn}

paths = {'dnn': 'model/dnn.h5',
         'cnn': 'model/cnn.h5',
         'rnn': 'model/rnn.h5'}

models = {'dnn': load_model('dnn', embed_mat, seq_len),
          'cnn': load_model('cnn', embed_mat, seq_len),
          'rnn': load_model('rnn', embed_mat, seq_len)}


def test_pair(name, pairs, flags, thre):
    model = map_item(name, models)
    sent1s, sent2s = pairs
    sims = model.predict([sent1s, sent2s])
    sims = np.reshape(sims, (1, -1))[0]
    preds = sims > thre
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(flags, preds)))
    for flag, sim, text1, text2, pred in zip(flags, sims, text1s, text2s, preds):
        if flag != pred:
            print('{} {:.3f} {} | {}'.format(flag, sim, text1, text2))


def test(name, texts, labels, vote):
    preds = list()
    for text in texts:
        preds.append(predict(text, name, vote))
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test_pair('dnn', pairs, flags, thre=0.5)
    test_pair('cnn', pairs, flags, thre=0.5)
    test_pair('rnn', pairs, flags, thre=0.5)
    test('dnn', texts, labels, vote=5)
    test('cnn', texts, labels, vote=5)
    test('rnn', texts, labels, vote=5)
