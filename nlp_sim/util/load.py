import pandas as pd

import numpy as np


def load_word(word):
    words = list()
    with open(word, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_sent(sent):
    sents = list()
    for num, text1, text2 in pd.read_csv(sent, header=None).values:
        sents.append(text1.strip())
        sents.append(text2.strip())
    return sents


def load_label(label):
    labels = list()
    with open(label, 'r') as f:
        for line in f:
            labels.append(int(line.strip()))
    return np.array(labels)


def load_sent_pair(sent):
    sent_pairs = list()
    for num, text1, text2 in pd.read_csv(sent, header=None).values:
        sent_pairs.append((text1.strip(), text2.strip()))
    return sent_pairs


def load_pred(pred):
    preds = list()
    with open(pred, 'r') as f:
        for line in f:
            pred = line.strip().split('\t')[1]
            preds.append(int(pred))
    return np.array(preds)
