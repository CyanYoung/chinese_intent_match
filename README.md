## Chinese Intent Match 2018-6

测试命令：bash run.sh nlp_sim/data/test.csv nlp_sim/data/test_pred.csv

#### 1.divide

train 70% / dev 20% / test 10% 划分，reindex() 重建索引、统计正例

#### 2.preprocess

delete() 去除无效符号，replace() 替换同音、同义词

jieba.load_userdict() 导入切分词，Counter() 统计词频

#### 3.vectorize

CountVectorizer() 过滤停用词、低频词得到每句的词频特征 bow

TfidfTransformer() 通过 bow 得到每句的词权特征 tfidf

Tokenizer() 建立词与索引的映射 word2ind

结合 word2ind 与 word2vec 得到 embed_mat、即 ind2vec

texts_to_sequences() 得到每句的词索引表示、pad_sequences() 填充为相同长度

#### 4.svm

subtract() 两句相减得到 diff 特征、multiply() 两句相乘得到 prod 特征

concat() 连接 diff 与 prod 得到 merge_features，SVC() 分类

#### 5.nn

dnn：average 算术平均、flat 展开，cnn：parallel 单层多核、serial 双层单核

rnn：plain 单层单向，stack 双层单向，bi 单层双向，attend 加权平均