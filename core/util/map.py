from core.nn_arch.dnn import dnn_siam_average
from core.nn_arch.cnn import cnn_siam_parallel, cnn_siam_serial
from core.nn_arch.rnn import rnn_siam_plain, rnn_siam_stack, rnn_siam_bi
from core.nn_arch.rnn import rnn_siam_attend, rnn_siam_bi_attend

from core.nn_arch.dnn import dnn_join_flat
from core.nn_arch.cnn import cnn_join_parallel, cnn_join_serial
from core.nn_arch.rnn import rnn_join_plain, rnn_join_stack, rnn_join_bi

from core.util.log import get_loggers


names = {'line': 'linear'}


loggers = {'svm': get_loggers('svm', 'core/log/svm/'),
           'dnn': get_loggers('dnn', 'core/log/dnn/'),
           'cnn': get_loggers('cnn', 'core/log/cnn/'),
           'rnn': get_loggers('rnn', 'core/log/rnn/')}


funcs = {'dnn_siam_average': dnn_siam_average,
         'cnn_siam_parallel': cnn_siam_parallel,
         'cnn_siam_serial': cnn_siam_serial,
         'rnn_siam_plain': rnn_siam_plain,
         'rnn_siam_stack': rnn_siam_stack,
         'rnn_siam_bi': rnn_siam_bi,
         'rnn_siam_attend': rnn_siam_attend,
         'rnn_siam_bi_attend': rnn_siam_bi_attend,
         'dnn_join_flat': dnn_join_flat,
         'cnn_join_parallel': cnn_join_parallel,
         'cnn_join_serial': cnn_join_serial,
         'rnn_join_plain': rnn_join_plain,
         'rnn_join_stack': rnn_join_stack,
         'rnn_join_bi': rnn_join_bi}


def map_name(name):
    if name in names:
        return names[name]
    else:
        return name


def map_logger(logger):
    if logger in loggers:
        return loggers[logger]
    else:
        raise KeyError


def map_func(func):
    if func in funcs:
        return funcs[func]
    else:
        raise KeyError
