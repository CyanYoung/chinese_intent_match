## Text Match 2018-6

测试命令：bash run.sh nlp_sim/data/test.csv nlp_sim/data/test_pred.csv

#### 1.divide

train 70% / dev 20% / test 10% 划分，reindex() 重建索引、统计正例

#### 2.preprocess

delete() 删除无效标点符 invalid_punc

replace() 替换同音词 homonym、同义词 synonym

jieba.load_userdict() 切分特殊词 special_word

Counter() 统计词频 vocab_freq，选出低频词 rare_word

#### 3.vectorize

合并停用词 stop_word 与 rare_word 为无效词 invalid_word

CountVectorizer() 过滤 invalid_word 得到每句的词频特征 bow

TfidfTransformer() 通过 bow 得到每句的词权特征 tfidf

Word2Vec() 设置 min_count 过滤低频词得到词向量 word2vec

Tokenizer() 建立词与索引的转换 word2ind

结合 word2ind 与 word2vec 得到 embed_mat、即 ind2vec

texts_to_sequences() 得到每句的词索引表示、pad_sequences() 填充为相同长度

#### 4.svm

subtract() 两句相减得到 diff 特征、multiply() 两句相乘得到 prod 特征

concat() 连接 diff 与 prod 得到 merge_features，SVC() 分类

#### 5.nn







