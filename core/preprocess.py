import pandas as pd

import re
import jieba

from collections import Counter


def delete(path_data, path_data_clean, path_inval_punc):
    reg = '[ '
    with open(path_inval_punc, 'r') as f:
        for line in f:
            reg = reg + line.strip()
    reg = reg + ']'
    with open(path_data, 'r') as f:
        data = f.read()
    data = re.sub(reg, '', data)
    with open(path_data_clean, 'w') as f:
        f.write(data)


def replace(path_data_clean, path_homonym, path_synonym):
    with open(path_data_clean, 'r') as f:
        data = f.read()
    for std, nstd in pd.read_csv(path_homonym).values:
        data = re.sub(nstd, std, data)
    for std, nstd in pd.read_csv(path_synonym).values:
        data = re.sub(nstd, std, data)
    with open(path_data_clean, 'w') as f:
        f.write(data)


def insert(field, texts, text_lens, vocabs, char):
    if char:
        words = list(field)
    else:
        words = list(jieba.cut(field))
    texts.append(' '.join(words))
    text_lens.append(len(words))
    vocabs.extend(words)


def count(path_freq, items):
    item_freq = Counter(items)
    with open(path_freq, 'w') as f:
        f.write('item,freq\n')
        for item, freq in item_freq.most_common():
            f.write(str(item) + ',' + str(freq) + '\n')


def preprocess(paths, mode, char):
    delete(paths['data'], paths['data_clean'], paths['inval_punc'])
    replace(paths['data_clean'], paths['homonym'], paths['synonym'])
    jieba.load_userdict(paths['cut_word'])
    nums, text1s, text2s = list(), list(), list()
    text_lens, vocabs = list(), list()
    with open(paths['data_clean'], 'r') as f:
        for line in f:
            num, text1, text2 = line.strip().split('\t')
            nums.append(num)
            insert(text1, text1s, text_lens, vocabs, char)
            insert(text2, text2s, text_lens, vocabs, char)
    with open(paths['data_clean'], 'w') as f:
        f.write('num,text1,text2\n')
        for num, text1, text2 in zip(nums, text1s, text2s):
            f.write(num + ',' + text1 + ',' + text2 + '\n')
    if mode == 'train':
        count(paths['len_freq'], text_lens)
        count(paths['vocab_freq'], vocabs)


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.csv'
    paths['data_clean'] = 'data/train_clean.csv'
    paths['inval_punc'] = 'dict/inval_punc.txt'
    paths['homonym'] = 'dict/homonym.csv'
    paths['synonym'] = 'dict/synonym.csv'
    paths['cut_word'] = 'dict/cut_word.txt'
    paths['len_freq'] = 'stat/len_freq.csv'
    paths['vocab_freq'] = 'stat/vocab_freq.csv'
    preprocess(paths, 'train', char=True)
    paths['data'] = 'data/dev.csv'
    paths['data_clean'] = 'data/dev_clean.csv'
    preprocess(paths, 'dev', char=True)
    paths['data'] = 'data/test.csv'
    paths['data_clean'] = 'data/test_clean.csv'
    preprocess(paths, 'test', char=True)
