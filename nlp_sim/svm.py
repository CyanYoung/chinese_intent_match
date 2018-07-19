import pickle as pk

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.svm import SVC

from nlp_sim.util.load import load_label
from nlp_sim.util.map import map_name, map_logger
from nlp_sim.util.trial import trial
from nlp_sim.util.log import log_state


def subtract(sent_features):
    sent_features = sent_features.toarray()
    feature_shape = sent_features.shape
    sent_diffs = np.zeros((int(feature_shape[0] / 2), feature_shape[1]))
    for i in range(len(sent_features) - 1):
        if not i % 2:
            sent_diffs[int(i / 2)] = np.abs(sent_features[i] - sent_features[i + 1])
    return sent_diffs


def multiply(sent_features):
    sent_features = sent_features.toarray()
    feature_shape = sent_features.shape
    sent_prods = np.zeros((int(feature_shape[0] / 2), feature_shape[1]))
    for i in range(len(sent_features) - 1):
        if not i % 2:
            sent_prods[int(i / 2)] = sent_features[i] * sent_features[i + 1]
    return sent_prods


def concat(sent_features):
    sent_features = sent_features.toarray()
    feature_shape = sent_features.shape
    sent_concats = np.zeros((int(feature_shape[0] / 2), feature_shape[1] * 2))
    for i in range(len(sent_features) - 1):
        if not i % 2:
            sent_concats[int(i / 2)] = np.hstack((sent_features[i], sent_features[i + 1]))
    return sent_concats


def svm(paths, kernel, feature, mode, thre):
    logger = map_logger('svm')
    name = '_'.join(['svm', kernel, feature])
    with open(paths[feature + '_feature'], 'rb') as f:
        sent_features = pk.load(f)
    diff = subtract(sent_features)
    prod = multiply(sent_features)
    merge_features = csr_matrix(np.hstack((diff, prod)))
    if mode == 'train':
        labels = load_label(paths['label'])
        kernel = map_name(kernel)
        model = SVC(C=1.0, kernel=kernel, probability=True, verbose=True)
        log_state(logger[0], name, mode)
        model.fit(merge_features, labels)
        log_state(logger[0], name, mode)
        with open(paths[name], 'wb') as f:
            pk.dump(model, f)
    elif mode == 'dev' or mode == 'test':
        labels = load_label(paths['label'])
        with open(paths[name], 'rb') as f:
            model = pk.load(f)
    else:
        raise KeyError
    probs = model.predict_proba(merge_features)[:, 1]
    if mode == 'train' or mode == 'dev':
        trial(paths['data_clean'], probs, labels, logger, name, mode)
    elif mode == 'test':
        return probs > thre
