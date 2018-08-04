import pandas as pd

import re
import jieba

from collections import Counter


min_freq = 5


def delete(data, data_clean, invalid_punc):
    reg = '[ '  # space
    with open(invalid_punc, 'r') as f:
        for line in f:
            reg = reg + line.strip()
    reg = reg + ']'
    with open(data, 'r') as f:
        data = f.read()
    data = re.sub(reg, '', data)
    with open(data_clean, 'w') as f:
        f.write(data)


def replace(data_clean, homonym, synonym):
    with open(data_clean, 'r') as f:
        data = f.read()
    for std, nstd in pd.read_csv(homonym).values:
        data = re.sub(nstd, std, data)
    for std, nstd in pd.read_csv(synonym).values:
        data = re.sub(nstd, std, data)
    with open(data_clean, 'w') as f:
        f.write(data)


def insert(field, texts, text_lens, vocabs, char):
    if char:
        words = list(field)
    else:
        words = list(jieba.cut(field))
    texts.append(' '.join(words))
    text_lens.append(len(words))
    vocabs.extend(words)


def preprocess(paths, mode, char):
    delete(paths['data'], paths['data_clean'], paths['invalid_punc'])
    replace(paths['data_clean'], paths['homonym'], paths['synonym'])
    jieba.load_userdict(paths['cut_word'])
    nums = list()
    text1s = list()
    text2s = list()
    text_lens = list()
    vocabs = list()
    with open(paths['data_clean'], 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            nums.append(fields[0])
            insert(fields[1], text1s, text_lens, vocabs, char)
            insert(fields[2], text2s, text_lens, vocabs, char)
    with open(paths['data_clean'], 'w') as f:
        for num, text1, text2 in zip(nums, text1s, text2s):
            f.write(num + ',' + text1 + ',' + text2 + '\n')
    if mode == 'train':
        vocab_count = Counter(vocabs)
        with open(paths['vocab_freq'], 'w') as fc:
            fc.write('vocab,freq' + '\n')
            with open(paths['rare_word'], 'w') as fr:
                for vocab, count in vocab_count.most_common():
                    fc.write(vocab + ',' + str(count) + '\n')
                    if count < min_freq:
                        fr.write(vocab + '\n')
        len_count = Counter(text_lens)
        with open(paths['len_freq'], 'w') as f:
            f.write('len,freq' + '\n')
            for text_len, freq in len_count.most_common():
                f.write(str(text_len) + ',' + str(freq) + '\n')


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.csv'
    paths['data_clean'] = 'data/train_clean.csv'
    paths['invalid_punc'] = 'dict/invalid_punc.txt'
    paths['homonym'] = 'dict/homonym.csv'
    paths['synonym'] = 'dict/synonym.csv'
    paths['cut_word'] = 'dict/cut_word.txt'
    paths['len_freq'] = 'dict/len_freq.csv'
    paths['vocab_freq'] = 'dict/vocab_freq.csv'
    paths['rare_word'] = 'dict/rare_word.txt'
    preprocess(paths, 'train', char=True)
    paths['data'] = 'data/dev.csv'
    paths['data_clean'] = 'data/dev_clean.csv'
    preprocess(paths, 'dev', char=True)
    paths['data'] = 'data/test.csv'
    paths['data_clean'] = 'data/test_clean.csv'
    preprocess(paths, 'test', char=True)
