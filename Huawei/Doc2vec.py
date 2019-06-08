# coding:utf-8
import jieba
import csv
import gensim
import numpy as np
from gensim.models.doc2vec import Doc2Vec,LabeledSentence

TaggededDocument = gensim.models.doc2vec.TaggedDocument

#通过gensim的doc2vec训练句向量

def get_datasest():
    with open(r"demo/news_yuliao.csv", 'r',encoding='utf-8') as cf: ##此处是获取你的训练集的地方，从一个文件中读取出来，里面的内容是一行一句话
        docs = cf.readlines()
    x_yuliao = []
    for i, text in enumerate(docs):
        ##如果是已经分好词的，不用再进行分词，直接按空格切分即可
        word=' '.join(jieba.cut(text.split('\n')[0]))
        word_list=word.split(' ')
        #word_list = ' '.join(jieba.cut(text.split('\n')[0])).encode('utf-8').split('')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_yuliao.append(document)
    return x_yuliao

def train(x_train, size=50, epoch_num=1): ##size 是你最终训练出的句子向量的维度，自己尝试着修改一下

    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('demo/model/ko_d2v.d2v') ##模型保存的位置

    return model_dm


def ceshi():
    model_dm = Doc2Vec.load("demo/model/ko_d2v.d2v")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    x_train=[]
    word_list =[]
    with open(r"demo/newsVec_train_process.csv", 'r', encoding='utf-8') as cf:
        docs = cf.readlines()
        print(len(docs))
    for i, text in enumerate(docs):
        ##如果是已经分好词的，不用再进行分词，直接按空格切分即可
        word=' '.join(jieba.cut(text.split('\n')[0])).split(' ')
        word_list.append(word)
    print(len(word_list))
    with open(r'demo/all_news_train.csv', 'w', newline='', encoding='utf-8') as wf:
        for i,val_text in enumerate(word_list,0):
            #print(i)
            inferred_vector_dm = model_dm.infer_vector(val_text)  ##得到文本的向量
            x_train.append(inferred_vector_dm)
            ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
            #print(inferred_vector_dm)
            writer = csv.writer(wf)
            writer.writerow(inferred_vector_dm)
    x_train=np.array(x_train)
    return x_train


if __name__ == '__main__':
    x_yuliaoku = get_datasest()
    model_dm = train(x_yuliaoku)
    doc_2_vec = ceshi()
    print (type(doc_2_vec))
    print(doc_2_vec.shape)

