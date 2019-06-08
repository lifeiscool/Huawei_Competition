----------------------------------------------------项目流程------------------------------------------------------------
预处理：                1. CSI300数据下载  →  2. 筛选（数据缺失严重）  →  3. 规范化  →  4. 聚类  →  5. 选类别代表（质心）
                                                                                                                 丨
                                                                                                                 ↓
第一步（爬数据）：                                                                                       6. 按具体股吧爬数据
                                                                                                                 丨
                                                                                                                 ↓
第二步（情感分析）：    10. 规范化时间长度   ←   9. CNN得出情感值[1,-1]     ←   8. 句向量（doc2vc）   ←    7.分词
                               丨
                               ↓
第三步（股票预测）：    11. 建模预测股票（数据/自编码特征+LSTM）


------------------------------------------------------目录及文件--------------------------------------------------------
目录
cluster:        聚类结果
data:           沪深300股票信息及各种csv
demo:           演示时用到的结果
lstm_input:     合并lstm输入所需的  股价特征、股评特征、时间文件，以及最终的lstm输入price_withdate.csv
models:         cnn网络模型
models_lstm:    lstm网络模型
results:        结果对比图


文件
1_downloadData.py:  获取沪深300股票信息，筛选得到455938条记录（203支×2246日），保存到filtered_data.csv
2_clustering.py:    对筛选后的股票进行聚类，结果保存到cluster文件夹
3_clawer.py:        爬取每一类的质心股票的股吧评论，保存到data/newsVec_train，演示时结果保存到demo/newsVec_train.csv
4_textprocess.py:   预处理股评，去掉非法符号和空格，保存到data/newsVec_train_process.csv，演示时保存到demo/newsVec_train_process.csv
5_doc2vec.py:       将股评信息转换为向量，保存到data/all_news_train.csv，演示时保存到demo/all_news_train.csv
6_newVec.py:        利用cnn学习股评向量特征，保存到data/selfcoding_feature.csv，演示时保存到demo/selfcoding_feature.csv
7_selfcoding.py:    利用自编码cnn学习股票价格特征，保存到data/feature.csv，演示时保存到demo/feature.csv
8_merge.py:         合并对应时间的    股票价格特征 和 股评特征，保存到price_withdate.csv
9_lstm.py:          构建lstm，以合并的股价特征和股评特征为输入，预测股票价格走势

filtered_data.csv:      筛选后的股票信息数据
getprice.py:            获得单支股票信息数据
Readme.txt:             Readme
安装说明.txt:           安装说明
开发文档.txt:           开发文档