import pickle as pk

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from util import flat_read


min_freq = 5

path_bow = 'model/svm/bow.pkl'
path_tfidf = 'model/svm/tfidf.pkl'


def bow(sents, path_bow, mode):
    if mode == 'train':
        model = CountVectorizer(token_pattern='\w', min_df=min_freq)
        model.fit(sents)
        with open(path_bow, 'wb') as f:
            pk.dump(model, f)
    else:
        with open(path_bow, 'rb') as f:
            model = pk.load(f)
    return model.transform(sents).toarray()


def tfidf(bow_sents, path_tfidf, mode):
    if mode == 'train':
        model = TfidfTransformer()
        model.fit(bow_sents)
        with open(path_tfidf, 'wb') as f:
            pk.dump(model, f)
    else:
        with open(path_tfidf, 'rb') as f:
            model = pk.load(f)
    return model.transform(bow_sents).toarray()


def merge(sents):
    bound = int(len(sents) / 2)
    sent1s, sent2s = sents[:bound], sents[bound:]
    diffs, prods = list(), list()
    for sent1, sent2 in zip(sent1s, sent2s):
        diffs.append(np.abs(sent1 - sent2))
        prods.append(sent1 * sent2)
    return csr_matrix(np.hstack((diffs, prods)))


def featurize(paths, mode):
    sent1s = flat_read(paths['data'], 'text1')
    sent2s = flat_read(paths['data'], 'text2')
    labels = flat_read(paths['data'], 'label')
    sents = sent1s + sent2s
    bow_sents = bow(sents, path_bow, mode)
    tfidf_sents = tfidf(bow_sents, path_tfidf, mode)
    bow_feats, tfidf_feats = merge(bow_sents), merge(tfidf_sents)
    labels = np.array(labels)
    with open(paths['bow_sent'], 'wb') as f:
        pk.dump(bow_feats, f)
    with open(paths['tfidf_sent'], 'wb') as f:
        pk.dump(tfidf_feats, f)
    with open(paths['label'], 'wb') as f:
        pk.dump(labels, f)


if __name__ == '__main__':
    paths = dict()
    prefix = 'feat/svm/'
    paths['data'] = 'data/train.csv'
    paths['bow_sent'] = prefix + 'bow_sent_train.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    featurize(paths, 'train')
    paths['data'] = 'data/test.csv'
    paths['bow_sent'] = prefix + 'bow_sent_test.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    featurize(paths, 'test')
