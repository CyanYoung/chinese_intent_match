import pickle as pk

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from nn_arch import dnn_siam, dnn_join, cnn_siam, cnn_join, rnn_siam, rnn_join

from util import map_item


batch_size = 128

path_embed = 'feat/embed.pkl'
path_pair = 'feat/pair_train.pkl'
path_label = 'feat/label_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

funcs = {'dnn_siam': dnn_siam,
         'dnn_join': dnn_join,
         'cnn_siam': cnn_siam,
         'cnn_join': cnn_join,
         'rnn_siam': rnn_siam,
         'rnn_join': rnn_join}

paths = {'dnn_siam': 'model/dnn_siam.h5',
         'dnn_join': 'model/dnn_join.h5',
         'cnn_siam': 'model/cnn_siam.h5',
         'cnn_join': 'model/cnn_join.h5',
         'rnn_siam': 'model/rnn_siam.h5',
         'rnn_join': 'model/rnn_join.h5',
         'dnn_siam_plot': 'model/plot/dnn_siam.png',
         'dnn_join_plot': 'model/plot/dnn_join.png',
         'cnn_siam_plot': 'model/plot/cnn_siam.png',
         'cnn_join_plot': 'model/plot/cnn_join.png',
         'rnn_siam_plot': 'model/plot/rnn_siam.png',
         'rnn_join_plot': 'model/plot/rnn_join.png'}


def compile(name, embed_mat, seq_len):
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


def fit(name, epoch, embed_mat, pairs, labels):
    sent1s, sent2s = pairs
    seq_len = len(sent1s[0])
    model = compile(name, embed_mat, seq_len)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit([sent1s, sent2s], labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    fit('dnn_siam', 10, embed_mat, pairs, labels)
    fit('dnn_join', 10, embed_mat, pairs, labels)
    fit('cnn_siam', 10, embed_mat, pairs, labels)
    fit('cnn_join', 10, embed_mat, pairs, labels)
    fit('rnn_siam', 10, embed_mat, pairs, labels)
    fit('rnn_join', 10, embed_mat, pairs, labels)
