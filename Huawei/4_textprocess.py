# -*-coding:utf-8-*-
import os
import sys
import jieba
from string import punctuation
import csv
import re

#转换文件文件格式
def transfer():
    path_0 = r"E:\项目\Huawei\news_sohusite_xml.smarty"

    path_1 = r"E:\项目\Huawei\news_sohusite_xml.smarty" + '\\'
    sys.path.append(path_1)
    files = os.listdir(path_0)
    print('files', files)
    for filename in files:
        portion = os.path.splitext(filename)
        if portion[1] == ".dat":
            # recombine file name
            newname = portion[0] + ".txt"
            filenamedir = path_1 + filename
            newnamedir = path_1 + newname
            # os.rename(filename, newname)
            os.rename(filenamedir, newnamedir)

#提取爬虫文件中的文本，并对文本进行处理，删除标点符号
add_punc = '，。、【】“”：；（）《》‘’{}＜＞!?/＝~★◥◣█◢ ▌▅▇█◤●！▃▃▁▂▃？一→￥▲一⑦()、%^>℃：.”“^-——=&#@$❤*＋▆．'
all_punc = punctuation + add_punc

with open('demo/newsVec_train.csv',encoding='utf-8') as rf,open(r'demo/newsVec_train_process.csv','w',newline='',encoding='utf-8') as wf:
    reader=csv.reader(rf)
    writer=csv.writer(wf)
    for row in reader:
        re=[]
        for i in row[2]:
            re.append(i)
            if i in add_punc:
                re.remove(i)
        sec=[''.join(re)]
        print(sec)
        writer.writerow(sec)




