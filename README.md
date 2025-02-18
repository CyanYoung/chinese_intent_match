## Chinese Intent Match 2018-5

#### 1.preprocess

clean() 去除停用词，替换同音、同义词，prepare() 打乱后划分训练、测试集

#### 2.explore

统计词汇、长度、类别的频率，条形图可视化，计算 sent / word_per_sent 指标

#### 3.featurize

ml 特征化，bow() 构建词袋模型，svd() 降维、平滑，merge() 合并 diff、prod

#### 4.vectorize

nn 向量化，embed() 建立词索引到词向量的映射，align() 将序列填充为相同长度

#### 5.build

ml_fit() 通过 svm、xgb，nn_fit() 通过 dnn、cnn_1d、cnn_2d、rnn 构建匹配模型

#### 6.match

predict() 实时交互，输入单句、经过清洗后预测，输出相似概率
