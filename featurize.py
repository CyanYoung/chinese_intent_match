import pickle as pk

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD

from util import flat_read


min_freq = 5

path_bow = 'model/ml/bow.pkl'
path_svd = 'model/ml/svd.pkl'


def sent2feat(sents, path_bow, path_svd, mode):
    if mode == 'train':
        bow = CountVectorizer(token_pattern='\w', min_df=min_freq)
        bow.fit(sents)
        with open(path_bow, 'wb') as f:
            pk.dump(bow, f)
    else:
        with open(path_bow, 'rb') as f:
            bow = pk.load(f)
    bow_sents = bow.transform(sents)
    if mode == 'train':
        svd = TruncatedSVD(n_components=200, n_iter=10)
        svd.fit(bow_sents)
        with open(path_svd, 'wb') as f:
            pk.dump(svd, f)
    else:
        with open(path_svd, 'rb') as f:
            svd = pk.load(f)
    return svd.transform(bow_sents)


def merge(sents):
    bound = int(len(sents) / 2)
    sent1s, sent2s = sents[:bound], sents[bound:]
    diffs, prods = list(), list()
    for sent1, sent2 in zip(sent1s, sent2s):
        diffs.append(np.abs(sent1 - sent2))
        prods.append(sent1 * sent2)
    return np.hstack((diffs, prods))


def featurize(path_data, path_sent, path_label, mode):
    sent1s = flat_read(path_data, 'text1')
    sent2s = flat_read(path_data, 'text2')
    labels = flat_read(path_data, 'label')
    sents = sent1s + sent2s
    sent_feats = sent2feat(sents, path_bow, path_svd, mode)
    sent_feats = merge(sent_feats)
    labels = np.array(labels)
    with open(path_sent, 'wb') as f:
        pk.dump(sent_feats, f)
    with open(path_label, 'wb') as f:
        pk.dump(labels, f)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/ml/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    featurize(path_data, path_sent, path_label, 'train')
    path_data = 'data/test.csv'
    path_sent = 'feat/ml/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    featurize(path_data, path_sent, path_label, 'test')
