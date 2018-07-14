from nlp_sim.svm import svm


if __name__ == '__main__':
    paths = dict()
    prefix = 'nlp_sim/'
    paths['data_clean'] = prefix + 'data/train_clean.csv'
    paths['label'] = prefix + 'data/train_label.txt'
    paths['bow_feature'] = prefix + 'feature/svm/bow_train.pkl'
    paths['tfidf_feature'] = prefix + 'feature/svm/tfidf_train.pkl'
    paths['svm_linear_bow'] = prefix + 'model/svm/linear_bow.pkl'
    paths['svm_linear_tfidf'] = prefix + 'model/svm/linear_tfidf.pkl'
    paths['svm_rbf_bow'] = prefix + 'model/svm/rbf_bow.pkl'
    svm(paths, 'linear', 'bow', 'train', thre=None)
    svm(paths, 'linear', 'tfidf', 'train', thre=None)
    svm(paths, 'rbf', 'bow', 'train', thre=None)
    paths['data_clean'] = prefix + 'data/dev_clean.csv'
    paths['label'] = prefix + 'data/dev_label.txt'
    paths['bow_feature'] = prefix + 'feature/svm/bow_dev.pkl'
    paths['tfidf_feature'] = prefix + 'feature/svm/tfidf_dev.pkl'
    svm(paths, 'linear', 'bow', 'dev', thre=None)
    svm(paths, 'linear', 'tfidf', 'dev', thre=None)
    svm(paths, 'rbf', 'bow', 'dev', thre=None)
