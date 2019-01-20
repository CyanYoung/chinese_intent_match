import sys

from core.preprocess import preprocess
from core.vectorize import vectorize
from core.svm import svm
from core.nn import nn


def test(paths, path_output, model):
    if model == 'dnn':
        preds = nn(paths, 'dnn', 'siam_mean', 10, 'test', thre=0.3)
    elif model == 'cnn':
        preds = nn(paths, 'cnn', 'siam_wide', 10, 'test', thre=0.2)
    elif model == 'rnn':
        preds = nn(paths, 'rnn', 'siam_plain', 10, 'test', thre=0.3)
    else:
        preds = svm(paths, 'rbf', 'bow', 'test', thre=0.2)
    with open(path_output, 'w') as f:
        for i in range(len(preds)):
            f.write(str(i + 1) + '\t' + str(int(preds[i])) + '\n')


if __name__ == '__main__':
    file, path_input, path_output = sys.argv
    paths = dict()
    prefix = 'core/'
    paths['data'] = path_input
    paths['data_clean'] = prefix + 'data/test_clean.csv'
    paths['invalid_punc'] = prefix + 'dict/invalid_punc.txt'
    paths['homo'] = prefix + 'dict/homo.csv'
    paths['syno'] = prefix + 'dict/syno.csv'
    paths['cut_word'] = prefix + 'dict/cut_word.txt'
    paths['vocab_freq'] = prefix + 'dict/vocab_freq.csv'
    paths['stop_word'] = prefix + 'dict/stop_word.txt'
    paths['bow_model'] = prefix + 'model/vec/bow.pkl'
    paths['tfidf_model'] = prefix + 'model/vec/tfidf.pkl'
    paths['bow_feat'] = prefix + 'feat/svm/bow_test.pkl'
    paths['tfidf_feat'] = prefix + 'feat/svm/tfidf_test.pkl'
    paths['word2ind'] = prefix + 'model/vec/word2ind.pkl'
    paths['pad'] = prefix + 'feat/nn/pad_test.pkl'
    paths['embed'] = prefix + 'feat/nn/embed.pkl'
    preprocess(paths, 'test', char=True)
    vectorize(paths, 'test')
    paths['svm_line_bow'] = prefix + 'model/svm/line_bow.pkl'
    paths['svm_line_tfidf'] = prefix + 'model/svm/line_tfidf.pkl'
    paths['svm_rbf_bow'] = prefix + 'model/svm/rbf_bow.pkl'
    paths['svm_rbf_tfidf'] = prefix + 'model/svm/rbf_tfidf.pkl'
    test(paths, path_output, 'svm')
    paths['dnn_siam_mean'] = prefix + 'model/dnn/siam_mean.h5'
    paths['dnn_join_flat'] = prefix + 'model/dnn/join_flat.h5'
    test(paths, path_output, 'dnn')
    paths['cnn_siam_wide'] = prefix + 'model/cnn/siam_wide.h5'
    paths['cnn_siam_deep'] = prefix + 'model/cnn/siam_deep.h5'
    paths['cnn_join_wide'] = prefix + 'model/cnn/join_wide.h5'
    paths['cnn_join_deep'] = prefix + 'model/cnn/join_deep.h5'
    test(paths, path_output, 'cnn')
    paths['rnn_siam_plain'] = prefix + 'model/rnn/siam_plain.h5'
    paths['rnn_siam_stack'] = prefix + 'model/rnn/siam_stack.h5'
    paths['rnn_siam_attend'] = prefix + 'model/rnn/siam_attend.h5'
    paths['rnn_join_plain'] = prefix + 'model/rnn/join_plain.h5'
    paths['rnn_join_stack'] = prefix + 'model/rnn/join_stack.h5'
    test(paths, path_output, 'rnn')
