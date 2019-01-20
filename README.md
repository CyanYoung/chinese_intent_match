## Chinese Intent Match 2018-6

测试命令：bash run.sh core/data/test.csv core/data/test_pred.csv

#### 1.divide

train 70% / dev 20% / test 10% 划分，reindex() 重建索引、统计正例

#### 2.preprocess

delete() 删去无用字符，replace() 替换同音、同义词，count() 统计词频、长度

#### 3.vectorize

CountVectorizer()、TfidfTransformer() 分别得到每句的词频、词权特征

embed() 通过 word_inds 与 word_vecs 建立词索引到词向量的映射

texts_to_sequences() 表示为词索引、pad_sequences() 填充为相同长度

#### 4.svm

连接 diff、prod 特征，分别使用 line、rbf，通过 SVC() 构建匹配模型

#### 5.nn

dnn_mean 算术平均、dnn_flat 展开，cnn_wide 单层多核、cnn_deep

多层单核，rnn_plain 单层单向、rnn_stack 双层单向、rnn_attend 加权平均