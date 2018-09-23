import pickle as pk

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.svm import SVC

from nlp_sim.util.load import load_label
from nlp_sim.util.map import map_name, map_logger
from nlp_sim.util.trial import trial
from nlp_sim.util.log import log_state


def subtract(sent_feats):
    sent_feats = sent_feats.toarray()
    feat_shape = sent_feats.shape
    sent_diffs = np.zeros((int(feat_shape[0] / 2), feat_shape[1]))
    for i in range(len(sent_feats) - 1):
        if not i % 2:
            sent_diffs[int(i / 2)] = np.abs(sent_feats[i] - sent_feats[i + 1])
    return sent_diffs


def multiply(sent_feats):
    sent_feats = sent_feats.toarray()
    feat_shape = sent_feats.shape
    sent_prods = np.zeros((int(feat_shape[0] / 2), feat_shape[1]))
    for i in range(len(sent_feats) - 1):
        if not i % 2:
            sent_prods[int(i / 2)] = sent_feats[i] * sent_feats[i + 1]
    return sent_prods


def concat(sent_feats):
    sent_feats = sent_feats.toarray()
    feat_shape = sent_feats.shape
    sent_concats = np.zeros((int(feat_shape[0] / 2), feat_shape[1] * 2))
    for i in range(len(sent_feats) - 1):
        if not i % 2:
            sent_concats[int(i / 2)] = np.hstack((sent_feats[i], sent_feats[i + 1]))
    return sent_concats


def svm(paths, kernel, feat, mode, thre=None):
    logger = map_logger('svm')
    name = '_'.join(['svm', kernel, feat])
    with open(paths[feat + '_feat'], 'rb') as f:
        sent_feats = pk.load(f)
    diff = subtract(sent_feats)
    prod = multiply(sent_feats)
    merge_feats = csr_matrix(np.hstack((diff, prod)))
    if mode == 'train':
        labels = load_label(paths['label'])
        kernel = map_name(kernel)
        model = SVC(C=100.0, kernel=kernel, probability=True, verbose=True)
        log_state(logger[0], name, mode)
        model.fit(merge_feats, labels)
        log_state(logger[0], name, mode)
        with open(paths[name], 'wb') as f:
            pk.dump(model, f)
    else:
        labels = load_label(paths['label'])
        with open(paths[name], 'rb') as f:
            model = pk.load(f)
    probs = model.predict_proba(merge_feats)[:, 1]
    if mode == 'train' or mode == 'dev':
        trial(paths['data_clean'], probs, labels, logger, name, mode)
    else:
        return probs > thre
