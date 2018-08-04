import sys

from nlp_sim.preprocess import preprocess
from nlp_sim.vectorize import vectorize
from nlp_sim.svm import svm
from nlp_sim.nn import nn


def test(paths, output, model):
    if model == 'svm':
        preds = svm(paths, 'rbf', 'bow', 'test', thre=0.2)
    elif model == 'dnn':
        preds = nn(paths, 'dnn', 'siam_average', 500, 'test', thre=0.3)
    elif model == 'cnn':
        preds = nn(paths, 'cnn', 'siam_parallel', 500, 'test', thre=0.2)
    elif model == 'rnn':
        preds = nn(paths, 'rnn', 'siam_plain', 500, 'test', thre=0.3)
    else:
        raise KeyError
    with open(output, 'w') as f:
        for i in range(len(preds)):
            f.write(str(i + 1) + '\t' + str(int(preds[i])) + '\n')


if __name__ == '__main__':
    file, input, output = sys.argv
    paths = dict()
    prefix = 'nlp_sim/'
    paths['data'] = input
    paths['data_clean'] = prefix + 'data/test_clean.csv'
    paths['invalid_punc'] = prefix + 'dict/invalid_punc.txt'
    paths['homonym'] = prefix + 'dict/homonym.csv'
    paths['synonym'] = prefix + 'dict/synonym.csv'
    paths['cut_word'] = prefix + 'dict/cut_word.txt'
    paths['vocab_freq'] = prefix + 'dict/vocab_freq.csv'
    paths['stop_word'] = prefix + 'dict/stop_word.txt'
    paths['rare_word'] = prefix + 'dict/rare_word.txt'
    paths['bow_model'] = prefix + 'model/vec/bow.pkl'
    paths['tfidf_model'] = prefix + 'model/vec/tfidf.pkl'
    paths['bow_feature'] = prefix + 'feature/svm/bow_test.pkl'
    paths['tfidf_feature'] = prefix + 'feature/svm/tfidf_test.pkl'
    paths['word2ind'] = prefix + 'model/vec/word2ind.pkl'
    paths['pad'] = prefix + 'feature/nn/pad_test.pkl'
    paths['embed'] = prefix + 'feature/nn/embed.pkl'
    preprocess(paths, 'test', char=True)
    vectorize(paths, 'test')
    paths['svm_linear_bow'] = prefix + 'model/svm/linear_bow.pkl'
    paths['svm_linear_tfidf'] = prefix + 'model/svm/linear_tfidf.pkl'
    paths['svm_rbf_bow'] = prefix + 'model/svm/rbf_bow.pkl'
    # test(paths, output, 'svm')
    paths['dnn_siam_average'] = prefix + 'model/dnn/siam_average.h5'
    paths['dnn_join_flat'] = prefix + 'model/dnn/join_flat.h5'
    # test(paths, output, 'dnn')
    paths['cnn_siam_parallel'] = prefix + 'model/cnn/siam_parallel.h5'
    paths['cnn_siam_serial'] = prefix + 'model/cnn/siam_serial.h5'
    paths['cnn_join_parallel'] = prefix + 'model/cnn/join_parallel.h5'
    paths['cnn_join_serial'] = prefix + 'model/cnn/join_serial.h5'
    # test(paths, output, 'cnn')
    paths['rnn_siam_plain'] = prefix + 'model/rnn/siam_plain.h5'
    paths['rnn_siam_stack'] = prefix + 'model/rnn/siam_stack.h5'
    paths['rnn_siam_attend'] = prefix + 'model/rnn/siam_attend.h5'
    paths['rnn_siam_bi_attend'] = prefix + 'model/rnn/siam_bi_attend.h5'
    test(paths, output, 'rnn')
