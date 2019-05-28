import pickle as pk

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from featurize import merge

from util import map_item


seq_len = 30

path_bow = 'model/ml/bow.pkl'
path_svm = 'model/ml/svm.pkl'
path_svd = 'model/ml/svd.pkl'
path_xgb = 'model/ml/xgb.pkl'
with open(path_bow, 'rb') as f:
    bow = pk.load(f)
with open(path_svd, 'rb') as f:
    svd = pk.load(f)
with open(path_svm, 'rb') as f:
    svm = pk.load(f)
with open(path_xgb, 'rb') as f:
    xgb = pk.load(f)

path_word2ind = 'model/nn/word2ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

paths = {'dnn': 'model/nn/dnn.h5',
         'cnn_1d': 'model/nn/cnn_1d.h5',
         'cnn_2d': 'model/nn/cnn_2d.h5',
         'rnn': 'model/nn/rnn.h5'}

models = {'svm': svm,
          'xgb': xgb,
          'dnn': load_model(map_item('dnn', paths)),
          'cnn_1d': load_model(map_item('cnn_1d', paths)),
          'cnn_2d': load_model(map_item('cnn_2d', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def ml_predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)
    text = [text1, text2]
    sent = bow.transform(text)
    sent = svd.transform(sent)
    sent = merge(sent)
    model = map_item(name, models)
    prob = model.predict_proba(sent)[0][1]
    return '{:.3f}'.format(prob)


def nn_predict(text1, text2, name):
    text1, text2 = clean(text1), clean(text2)
    seq1 = word2ind.texts_to_sequences([text1])[0]
    seq2 = word2ind.texts_to_sequences([text2])[0]
    pad_seq1 = pad_sequences([seq1], maxlen=seq_len)
    pad_seq2 = pad_sequences([seq2], maxlen=seq_len)
    model = map_item(name, models)
    prob = model.predict([pad_seq1, pad_seq2])[0][0]
    return '{:.3f}'.format(prob)


if __name__ == '__main__':
    while True:
        text1, text2 = input('text1: '), input('text2: ')
        print('svm: %s' % ml_predict(text1, text2, 'svm'))
        print('xgb: %s' % ml_predict(text1, text2, 'xgb'))
        print('dnn: %s' % nn_predict(text1, text2, 'dnn'))
        print('cnn_1d: %s' % nn_predict(text1, text2, 'cnn_1d'))
        print('cnn_2d: %s' % nn_predict(text1, text2, 'cnn_2d'))
        print('rnn: %s' % nn_predict(text1, text2, 'rnn'))
