import re

from random import shuffle

from util import load_word_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)


def save(path, pairs):
    head = 'text1,text2,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text1, text2, label in pairs:
            f.write(text1 + ',' + text2 + ',' + str(label) + '\n')


def clean(text):
    text = re.sub(stop_word_re, '', text.strip())
    text = word_replace(text, homo_dict)
    return word_replace(text, syno_dict)


def prepare(path_univ, path_train, path_test):
    pairs = list()
    with open(path_univ, 'r') as f:
        for line in f:
            num, text1, text2, label = line.strip().split('\t')
            text1, text2 = clean(text1), clean(text2)
            pairs.append((text1, text2, label))
    shuffle(pairs)
    bound = int(len(pairs) * 0.9)
    save(path_train, pairs[:bound])
    save(path_test, pairs[bound:])


if __name__ == '__main__':
    path_univ = 'data/univ.csv'
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    prepare(path_univ, path_train, path_test)
