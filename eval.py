import pickle as pk

from sklearn.metrics import f1_score, accuracy_score

from match import models

from util import map_item


path_pair = 'feat/pair_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, pairs, labels, thre):
    model = map_item(name, models)
    sent1s, sent2s = pairs
    probs = model.predict([sent1s, sent2s])
    preds = probs > thre
    print('\n%s f1: %.2f - acc: %.2f' % (name, f1_score(labels, preds),
                                         accuracy_score(labels, preds)))


if __name__ == '__main__':
    test('dnn', pairs, labels, thre=0.5)
    test('cnn_1d', pairs, labels, thre=0.5)
    test('cnn_2d', pairs, labels, thre=0.5)
    test('rnn', pairs, labels, thre=0.5)
