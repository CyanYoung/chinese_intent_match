import logging
import logging.handlers as handlers

from sklearn.metrics import f1_score, accuracy_score


def get_loggers(name, prefix):
    loggers = list()
    loggers.append(set_logger(name + '0', prefix + 'score.log', time=True))
    loggers.append(set_logger(name + '1', prefix + 'score.log', time=False))
    loggers.append(set_logger(name + '2', prefix + 'error.log', time=True))
    loggers.append(set_logger(name + '3', prefix + 'error.log', time=False))
    return loggers


def set_logger(name, path, time):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = handlers.RotatingFileHandler(path, 'a', 0, 1)
    fh.setLevel(logging.INFO)
    if time:
        formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    else:
        formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def log_state(logger, name, mode):
    logger.info(' '.join([name, mode]))


def log_score(logger, labels, preds, name, mode):
    logger.info('{} {} f1  {:.3f}'.format(name, mode, f1_score(labels, preds)))
    logger.info('{} {} acc {:.3f}'.format(name, mode, accuracy_score(labels, preds)))
    logger.info('')


def log_error(logger, labels, preds, probs, sent_pairs):
    logger.info('')
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            logger.info('{} {:.3f} {} | {}'.format(labels[i], probs[i], sent_pairs[i][0], sent_pairs[i][1]))
    logger.info('')
