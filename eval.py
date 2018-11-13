from sklearn.metrics import f1_score, accuracy_score

from core.util.load import load_label, load_pred


if __name__ == '__main__':
    paths = dict()
    prefix = 'core/'
    paths['label'] = prefix + 'data/test_label.txt'
    paths['pred'] = prefix + 'data/test_pred.csv'
    labels = load_label(paths['label'])
    preds = load_pred(paths['pred'])
    print('test f1  {:.3f}'.format(f1_score(labels, preds)))
    print('test acc {:.3f}'.format(accuracy_score(labels, preds)))
