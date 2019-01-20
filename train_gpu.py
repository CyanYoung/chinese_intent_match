from core.nn import nn


def siam(paths):
    paths['dnn_siam_mean'] = prefix + 'model/dnn/siam_mean.h5'
    paths['cnn_siam_wide'] = prefix + 'model/cnn/siam_wide.h5'
    paths['cnn_siam_deep'] = prefix + 'model/cnn/siam_deep.h5'
    paths['rnn_siam_plain'] = prefix + 'model/rnn/siam_plain.h5'
    paths['rnn_siam_stack'] = prefix + 'model/rnn/siam_stack.h5'
    paths['rnn_siam_attend'] = prefix + 'model/rnn/siam_attend.h5'
    nn(paths, 'dnn', 'siam_mean', 10, 'train')
    nn(paths, 'dnn', 'siam_mean', 10, 'dev')
    nn(paths, 'cnn', 'siam_wide', 10, 'train')
    nn(paths, 'cnn', 'siam_wide', 10, 'dev')
    nn(paths, 'cnn', 'siam_deep', 10, 'train')
    nn(paths, 'cnn', 'siam_deep', 10, 'dev')
    nn(paths, 'rnn', 'siam_plain', 10, 'train')
    nn(paths, 'rnn', 'siam_plain', 10, 'dev')
    nn(paths, 'rnn', 'siam_stack', 10, 'train')
    nn(paths, 'rnn', 'siam_stack', 10, 'dev')
    nn(paths, 'rnn', 'siam_attend', 10, 'train')
    nn(paths, 'rnn', 'siam_attend', 10, 'dev')


def join(paths):
    paths['dnn_join_flat'] = prefix + 'model/dnn/join_flat.h5'
    paths['cnn_join_wide'] = prefix + 'model/cnn/join_wide.h5'
    paths['cnn_join_deep'] = prefix + 'model/cnn/join_deep.h5'
    paths['rnn_join_plain'] = prefix + 'model/rnn/join_plain.h5'
    paths['rnn_join_stack'] = prefix + 'model/rnn/join_stack.h5'
    nn(paths, 'dnn', 'join_flat', 10, 'train')
    nn(paths, 'dnn', 'join_flat', 10, 'dev')
    nn(paths, 'cnn', 'join_wide', 10, 'train')
    nn(paths, 'cnn', 'join_wide', 10, 'dev')
    nn(paths, 'cnn', 'join_deep', 10, 'train')
    nn(paths, 'cnn', 'join_deep', 10, 'dev')
    nn(paths, 'rnn', 'join_plain', 10, 'train')
    nn(paths, 'rnn', 'join_plain', 10, 'dev')
    nn(paths, 'rnn', 'join_stack', 10, 'train')
    nn(paths, 'rnn', 'join_stack', 10, 'dev')


if __name__ == '__main__':
    paths = dict()
    prefix = 'core/'
    paths['train_clean'] = prefix + 'data/train_clean.csv'
    paths['dev_clean'] = prefix + 'data/dev_clean.csv'
    paths['pad_train'] = prefix + 'feat/nn/pad_train.pkl'
    paths['pad_dev'] = prefix + 'feat/nn/pad_dev.pkl'
    paths['train_label'] = prefix + 'data/train_label.txt'
    paths['dev_label'] = prefix + 'data/dev_label.txt'
    paths['embed'] = prefix + 'feat/nn/embed.pkl'
    siam(paths)
    join(paths)
