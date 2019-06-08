import requests
from bs4 import BeautifulSoup
import time
import csv
import re
import codecs
#爬虫获取股吧评论
# 复制请求头
head = {
    'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'close',
        'Cookie': 'em_hq_fls=js; HAList=f-0-399300-%u6CAA%u6DF1300%2Cf-0-000001-%u4E0A%u8BC1%u6307%u6570; st_si=44473506354444; emshistory=%5B%22%E4%B8%AD%E8%81%94%E9%87%8D%E7%A7%91%22%2C%22%E6%81%92%E9%80%B8%E7%9F%B3%E5%8C%96%22%2C%22%E5%AE%89%E9%81%93%E9%BA%A6A%22%2C%22%E4%B8%9C%E6%97%AD%E5%85%89%E7%94%B5%22%5D; qgqp_b_id=4c190c12e0bb3ec95717a1fed7cf001a; st_pvi=22537593272243; st_sp=2018-10-14%2019%3A32%3A17; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=1055; st_psi=20190507210423902-117001301474-4922934312; st_asi=delete',
        'Host': 'djdk.eastmoney.com',
        'Referer': 'http://guba.eastmoney.com/list,601628,f.html',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36'}

# 设置数据存储方式，csv表格写入
f = open('demo/newsVec_train.csv', 'a', newline='',encoding='utf-8-sig')

w = csv.writer(f)

#获取帖子详细时间，列表也没有年份，可以作为获取帖子其他详细内容的通用方法
def get_time(url):
    try:
        q = requests.get(url,headers=head)
        soup = BeautifulSoup(q.text,'html.parser')
        ptime = soup.find('div',{'class':'l5 a5'}).get_text()
        ptime = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',ptime)[0]
        print(ptime)
        return ptime
    except:
        return ''

#获取列表页第n页的具体目标信息，由BeautifulSoup解析完成
def get_urls(url):
    print(url)
    baseurl = 'http://guba.eastmoney.com/'
    q = requests.get(url)
    q.encoding=q.apparent_encoding
    soup = BeautifulSoup(q.content,'lxml')
    urllist = soup.find_all(class_ = {'articleh ','articleh normal_post'})
    print(len(urllist))
    for i in urllist:
        if i.find('a') != None:
            try:
                detailurl = i.find('a').attrs['href'].replace('/','')
                #print(detailurl)
                titel = i.find('a').get_text()
                yuedu = i.find('span',{'class':'l1'}).get_text()
                pinlun = i.find('span', {'class': 'l2'}).get_text()
                timeurl=i.find('span', {'class': 'l3'}).get_text()
                author= i.find('span', {'class': 'l4'}).get_text()
                ttime=i.find('span', {'class': 'l5'}).get_text()
                ptime = get_time(baseurl+detailurl)
                w.writerow([yuedu,pinlun,titel,author,ttime])
                print(yuedu,pinlun,titel,author,ttime)
                #print(baseurl + detailurl)
            except:
                pass
for i in range(1,2):
    print(i)
    get_urls('http://guba.eastmoney.com/list,601628,f_'+str(i)+'.html')

