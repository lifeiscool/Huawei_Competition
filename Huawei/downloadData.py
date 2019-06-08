import baostock as bs
import pandas as pd
import numpy as np
import csv
import os

use_count = 0
filter_count = 0
row_count = 2246
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

def get_data(code):
    rs = bs.query_history_k_data(code,
    "date,code,open,close,high,low,volume,amount,turn",
    start_date='2010-01-01', end_date='2019-04-01',frequency="d", adjustflag="3")
    print('query_history_k_data respond error_code:'+rs.error_code)
    print('query_history_k_data respond  error_msg:'+rs.error_msg)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    #result.drop(['adjustflag'],axis=1,inplace=True)
    result.to_csv(str("datas/"+code+"_data.csv"), index=False,)
    # print(result)
    return None


def get_code(file_path):
    rs = bs.query_hs300_stocks()
    print('query_hs300 error_code:' + rs.error_code)
    print('query_hs300  error_msg:' + rs.error_msg)
    hs300_stocks = []
    while (rs.error_code == '0') & rs.next():
        hs300_stocks.append(rs.get_row_data())
    result = pd.DataFrame(hs300_stocks, columns=rs.fields)
    result.to_csv(file_path, encoding="utf-8", index=False)
    return None

def get_count(file_path,name,path):
    global use_count, row_count
    with open(file_path, "r", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        rows = rows[1:len(rows)]

        temp_col = np.array(rows)
        temp_count = [name]
        for i in range(len(rows[0])):
            count = 0
            for index in range(len(rows)):
                if(temp_col[:,i][index]==""):
                    count=count+1
            temp_count.append(count)
        temp_count.append(len(rows))
        row_count = len(rows) if len(rows)>=row_count else row_count
        if(len(rows)<1000):
            use_count = use_count+1
        print(use_count)
    with open(path, "a+", newline='') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        csv_file.writerow(temp_count)
    return None

def filter(files_path,save_path,loss_path):
    global row_count
    list = os.listdir(files_path)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(files_path, list[i])
        # print(path)
        with open(path, "r", encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
            rows = rows[1:len(rows)]
            # print(len(rows))
            if len(rows)>=row_count:
                with open(save_path, "a+", newline='') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
                    csv_file = csv.writer(file)
                    temp_col = np.array(rows)
                    csv_file.writerows(temp_col[:,0:6])
            else:
                with open(loss_path, "a+", newline='') as file:  # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
                    csv_file = csv.writer(file)
                    csv_file.writerow(list[i][0:-9])
    print("filter() success")
    return None



if __name__ == "__main__":
      #获取沪深300股票代码
    save_path = "hs300_stocks.csv"
    get_code(save_path)

      #获取股票价格数据：每支股票2010年至今信息
    with open("hs300_stocks.csv", "r",encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        for index in range(1,len(rows)):
            print(rows[index][1])
            get_data(rows[index][1])

      #获取数据中个属性缺失数量
    save_path = "loss_count.csv"
    if(os.path.exists(save_path)):
        os.remove(save_path)
    rootdir = 'datas'
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        get_count(path,list[i][3:-9],save_path)
    print(row_count)

      #获取筛选后数据
    files_path = "datas/"
    save_path = "filtered_data.csv"
    loss_path = "loss_data.csv"
    if (os.path.exists(save_path)):
        os.remove(save_path)
    if (os.path.exists(loss_path)):
        os.remove(loss_path)
    filter(files_path, save_path,loss_path)

    None
bs.logout()