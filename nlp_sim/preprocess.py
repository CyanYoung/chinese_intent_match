import pandas as pd

import re
import jieba

from collections import Counter


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
    for std, nstd in pd.read_csv(homonym, header=None).values:
        data = re.sub(nstd, std, data)
    for std, nstd in pd.read_csv(synonym, header=None).values:
        data = re.sub(nstd, std, data)
    with open(data_clean, 'w') as f:
        f.write(data)


def insert(field, texts, all_words):
    words = list(jieba.cut(field))
    texts.append(' '.join(words))
    all_words.extend(words)


def preprocess(paths, mode):
    delete(paths['data'], paths['data_clean'], paths['invalid_punc'])
    replace(paths['data_clean'], paths['homonym'], paths['synonym'])
    jieba.load_userdict(paths['special_word'])
    all_words = list()
    nums = list()
    text1s = list()
    text2s = list()
    with open(paths['data_clean'], 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            nums.append(fields[0])
            insert(fields[1], text1s, all_words)
            insert(fields[2], text2s, all_words)
    with open(paths['data_cut'], 'w') as f:
        for num, text1, text2 in zip(nums, text1s, text2s):
            f.write(num + ',' + text1 + ',' + text2 + '\n')
    if mode == 'train':
        counter = Counter(all_words)
        with open(paths['vocab_freq'], 'w') as fc:
            with open(paths['rare_word'], 'w') as fr:
                for vocab, freq in counter.most_common():
                    fc.write(vocab + ',' + str(freq) + '\n')
                    if freq < 4:
                        fr.write(vocab + '\n')


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.csv'
    paths['data_clean'] = 'data/train_clean.csv'
    paths['data_cut'] = 'data/train_cut.csv'
    paths['invalid_punc'] = 'dict/invalid_punc.txt'
    paths['homonym'] = 'dict/homonym.csv'
    paths['synonym'] = 'dict/synonym.csv'
    paths['special_word'] = 'dict/special_word.txt'
    paths['vocab_freq'] = 'dict/vocab_freq.csv'
    paths['rare_word'] = 'dict/rare_word.txt'
    preprocess(paths, 'train')
    paths['data'] = 'data/dev.csv'
    paths['data_clean'] = 'data/dev_clean.csv'
    paths['data_cut'] = 'data/dev_cut.csv'
    preprocess(paths, 'dev')
    # paths['data'] = 'data/test.csv'
    # paths['data_clean'] = 'data/test_clean.csv'
    # paths['data_cut'] = 'data/test_cut.csv'
    # preprocess(paths, 'test')
