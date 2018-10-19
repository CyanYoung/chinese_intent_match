from random import shuffle


def reindex(path_data, path_label, lines, mode):
    pos = 0
    with open(path_data, 'w') as fd:
        with open(path_label, 'w') as fl:
            for ind, line in enumerate(lines):
                num, text1, text2, label = line.strip().split('\t')
                num = str(ind + 1)
                fd.write('\t'.join([num, text1, text2]) + '\n')
                fl.write(label)
                pos = pos + int(label)
    print('{:<5} pos {:>5} rate {:.3f}'.format(mode, pos, pos / len(lines)))


def divide(paths):
    with open(paths['univ1'], 'r') as f:
        line1s = f.readlines()
    with open(paths['univ2'], 'r') as f:
        line2s = f.readlines()
    lines = line1s + line2s
    shuffle(lines)
    bound1 = int(len(lines) * 0.7)
    bound2 = int(len(lines) * 0.9)
    reindex(paths['train'], paths['train_label'], lines[:bound1], 'train')
    reindex(paths['dev'], paths['dev_label'], lines[bound1:bound2], 'dev')
    reindex(paths['test'], paths['test_label'], lines[bound2:], 'test')


if __name__ == '__main__':
    paths = dict()
    paths['univ1'] = 'data/univ1.csv'
    paths['univ2'] = 'data/univ2.csv'
    paths['train'] = 'data/train.csv'
    paths['train_label'] = 'data/train_label.txt'
    paths['dev'] = 'data/dev.csv'
    paths['dev_label'] = 'data/dev_label.txt'
    paths['test'] = 'data/test.csv'
    paths['test_label'] = 'data/test_label.txt'
    divide(paths)
