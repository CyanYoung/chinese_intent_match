import pickle as pk

from sklearn.svm import SVC

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from nn_arch import dnn, cnn_1d, cnn_2d, rnn

from util import map_item


batch_size = 128

path_bow_sent = 'feat/ml/bow_sent_train.pkl'
path_tfidf_sent = 'feat/ml/tfidf_sent_train.pkl'
with open(path_bow_sent, 'rb') as f:
    bow_sents = pk.load(f)
with open(path_tfidf_sent, 'rb') as f:
    tfidf_sents = pk.load(f)

path_embed = 'feat/nn/embed.pkl'
path_pair = 'feat/nn/pair_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)

path_label = 'feat/label_train.pkl'
with open(path_label, 'rb') as f:
    labels = pk.load(f)

feats = {'bow': bow_sents,
         'tfidf': tfidf_sents}

funcs = {'dnn': dnn,
         'cnn_1d': cnn_1d,
         'cnn_2d': cnn_2d,
         'rnn': rnn}

paths = {'svm': 'model/ml/svm.pkl',
         'dnn': 'model/nn/dnn.h5',
         'cnn_1d': 'model/nn/cnn_1d.h5',
         'cnn_2d': 'model/nn/cnn_2d.h5',
         'rnn': 'model/nn/rnn.h5',
         'dnn_plot': 'model/nn/plot/dnn.png',
         'cnn_1d_plot': 'model/nn/plot/cnn_1d.png',
         'cnn_2d_plot': 'model/nn/plot/cnn_2d.png',
         'rnn_plot': 'model/nn/plot/rnn.png'}


def svm_fit(kernel, feat, labels):
    sents = map_item(feat, feats)
    model = SVC(C=10.0, kernel=kernel, max_iter=-1, probability=True,
                class_weight='balanced', verbose=True)
    model.fit(sents, labels)
    with open(map_item('svm', paths), 'wb') as f:
        pk.dump(model, f)


def nn_compile(name, embed_mat, seq_len):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True, name='embed')
    input1 = Input(shape=(seq_len,))
    input2 = Input(shape=(seq_len,))
    embed_input1 = embed(input1)
    embed_input2 = embed(input2)
    func = map_item(name, funcs)
    output = func(embed_input1, embed_input2)
    model = Model([input1, input2], output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def nn_fit(name, epoch, embed_mat, pairs, labels):
    sent1s, sent2s = pairs
    seq_len = len(sent1s[0])
    model = nn_compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit([sent1s, sent2s], labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    svm_fit('rbf', 'bow', labels)
    nn_fit('dnn', 10, embed_mat, pairs, labels)
    nn_fit('cnn_1d', 10, embed_mat, pairs, labels)
    nn_fit('cnn_2d', 10, embed_mat, pairs, labels)
    nn_fit('rnn', 10, embed_mat, pairs, labels)
