import pickle as pk

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from util import flat_read


min_freq = 3

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
    return model.transform(sents)


def tfidf(bow_sents, path_tfidf, mode):
    if mode == 'train':
        model = TfidfTransformer()
        model.fit(bow_sents)
        with open(path_tfidf, 'wb') as f:
            pk.dump(model, f)
    else:
        with open(path_tfidf, 'rb') as f:
            model = pk.load(f)
    return model.transform(bow_sents)


def get_diff(sent_feats):
    sent_feats = sent_feats.toarray()
    feat_shape = sent_feats.shape
    sent_diffs = np.zeros((int(feat_shape[0] / 2), feat_shape[1]))
    for i in range(len(sent_feats) - 1):
        if not i % 2:
            sent_diffs[int(i / 2)] = np.abs(sent_feats[i] - sent_feats[i + 1])
    return sent_diffs


def get_prod(sent_feats):
    sent_feats = sent_feats.toarray()
    feat_shape = sent_feats.shape
    sent_prods = np.zeros((int(feat_shape[0] / 2), feat_shape[1]))
    for i in range(len(sent_feats) - 1):
        if not i % 2:
            sent_prods[int(i / 2)] = sent_feats[i] * sent_feats[i + 1]
    return sent_prods


def featurize(paths, mode):
    sent1s = flat_read(paths['data'], 'text1')
    sent2s = flat_read(paths['data'], 'text2')
    labels = flat_read(paths['data'], 'label')
    sents = sent1s + sent2s
    bow_sents = bow(sents, path_bow, mode)
    tfidf_sents = tfidf(bow_sents, path_tfidf, mode)
    labels = np.array(labels)
    with open(paths['bow_sent'], 'wb') as f:
        pk.dump(bow_sents, f)
    with open(paths['tfidf_sent'], 'wb') as f:
        pk.dump(tfidf_sents, f)
    with open(paths['label'], 'wb') as f:
        pk.dump(labels, f)


if __name__ == '__main__':
    paths = dict()
    prefix = 'feat/svm/'
    paths['data'] = 'data/train.csv'
    paths['bow_sent'] = prefix + 'bow_sent_train.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_train.pkl'
    paths['label'] = prefix + 'label_train.pkl'
    featurize(paths, 'train')
    paths['data'] = 'data/train.csv'
    paths['bow_sent'] = prefix + 'bow_sent_test.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_test.pkl'
    paths['label'] = prefix + 'label_test.pkl'
    featurize(paths, 'test')
