import numpy as np

from sklearn.metrics import f1_score

from nlp_sim.util.load import load_sent_pair
from nlp_sim.util.log import log_state, log_score, log_error


def trial(sent, probs, labels, logger, name, mode):
    sent_pairs = load_sent_pair(sent)
    thres = np.linspace(1, 9, 9) / 10
    pred_mat = np.zeros((len(thres), len(probs)))
    f1s = np.zeros(len(thres))
    for i in range(len(thres)):
        pred_mat[i] = probs > thres[i]
        f1s[i] = f1_score(labels, pred_mat[i])
    max_ind = np.argmax(f1s)
    thre = thres[max_ind]
    preds = pred_mat[max_ind]
    name = '_'.join([name, str(thre)])
    log_score(logger[1], labels, preds, name, mode)
    if mode == 'dev':
        log_state(logger[2], name, mode)
        log_error(logger[3], labels, preds, probs, sent_pairs)
