from nlp_sim.nn import nn


def siam(paths):
    paths['dnn_siam_average'] = prefix + 'model/dnn/siam_average.h5'
    paths['cnn_siam_parallel'] = prefix + 'model/cnn/siam_parallel.h5'
    paths['cnn_siam_serial'] = prefix + 'model/cnn/siam_serial.h5'
    paths['rnn_siam_plain'] = prefix + 'model/rnn/siam_plain.h5'
    paths['rnn_siam_stack'] = prefix + 'model/rnn/siam_stack.h5'
    paths['rnn_siam_attend'] = prefix + 'model/rnn/siam_attend.h5'
    paths['rnn_siam_bi_attend'] = prefix + 'model/rnn/siam_bi_attend.h5'
    nn(paths, 'dnn', 'siam_average', 100, 'train', thre=None)
    nn(paths, 'dnn', 'siam_average', 100, 'dev', thre=None)
    nn(paths, 'cnn', 'siam_parallel', 100, 'train', thre=None)
    nn(paths, 'cnn', 'siam_parallel', 100, 'dev', thre=None)
    nn(paths, 'cnn', 'siam_serial', 100, 'train', thre=None)
    nn(paths, 'cnn', 'siam_serial', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'siam_plain', 100, 'train', thre=None)
    nn(paths, 'rnn', 'siam_plain', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'siam_stack', 100, 'train', thre=None)
    nn(paths, 'rnn', 'siam_stack', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'siam_attend', 100, 'train', thre=None)
    nn(paths, 'rnn', 'siam_attend', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'siam_bi_attend', 100, 'train', thre=None)
    nn(paths, 'rnn', 'siam_bi_attend', 100, 'dev', thre=None)


def join(paths):
    paths['dnn_join_flat'] = prefix + 'model/dnn/join_flat.h5'
    paths['cnn_join_parallel'] = prefix + 'model/cnn/join_parallel.h5'
    paths['cnn_join_serial'] = prefix + 'model/cnn/join_serial.h5'
    paths['rnn_join_plain'] = prefix + 'model/rnn/join_plain.h5'
    paths['rnn_join_stack'] = prefix + 'model/rnn/join_stack.h5'
    nn(paths, 'dnn', 'join_flat', 100, 'train', thre=None)
    nn(paths, 'dnn', 'join_flat', 100, 'dev', thre=None)
    nn(paths, 'cnn', 'join_parallel', 100, 'train', thre=None)
    nn(paths, 'cnn', 'join_parallel', 100, 'dev', thre=None)
    nn(paths, 'cnn', 'join_serial', 100, 'train', thre=None)
    nn(paths, 'cnn', 'join_serial', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'join_plain', 100, 'train', thre=None)
    nn(paths, 'rnn', 'join_plain', 100, 'dev', thre=None)
    nn(paths, 'rnn', 'join_stack', 100, 'train', thre=None)
    nn(paths, 'rnn', 'join_stack', 100, 'dev', thre=None)


if __name__ == '__main__':
    paths = dict()
    prefix = 'nlp_sim/'
    paths['train_cut'] = prefix + 'data/train_cut.csv'
    paths['dev_cut'] = prefix + 'data/dev_cut.csv'
    paths['pad_train'] = prefix + 'feature/pad_train.pkl'
    paths['pad_dev'] = prefix + 'feature/pad_dev.pkl'
    paths['train_label'] = prefix + 'data/train_label.txt'
    paths['dev_label'] = prefix + 'data/dev_label.txt'
    paths['embed'] = prefix + 'feature/embed.pkl'
    siam(paths)
    join(paths)
