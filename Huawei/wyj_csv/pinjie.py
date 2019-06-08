# coding=utf-8
import csv
import os

def get_unified(price_path,time_path,unified_path,save_path,num_index,date_index):
    """
    param:
        csv_path : csv文件目录及文件名
        num_index : 数字所在列
        date_index : 日期所在列
        results_path : 合并后的结果
    return:

    e.g.
        csv_path = os.path.join("unified_scv.csv")
        num_index = 2
        date_index = 0
        results_path = os.path.join("unified_results.csv")
    """

    #读取csv
    csv_file = csv.reader(open(price_path, "r"))
    rows = [row for row in csv_file]
    time_file = csv.reader(open(time_path, "r"))
    time_rows = [row for row in time_file]
    
    unified_file = csv.reader(open(unified_path, "r"))
    unified_rows = [row for row in unified_file]

    save_rows = []

    temp_price = 0
    temp_price2 = 0
    for i in range(1,len(unified_rows)):
        flag = 1
        for j in range(1,len(rows)):
            if(time_rows[j][0] == unified_rows[i][1]):
                if(j==len(rows)-1):
                    save_rows.append([unified_rows[i][1], unified_rows[i][0], rows[j][0][1:-1], rows[j][0][1:-1]])
                    temp_price = rows[j][0][1:-1]
                    temp_price2 = rows[j][0][1:-1]
                else:
                    save_rows.append([unified_rows[i][1],unified_rows[i][0], rows[j][0][1:-1],rows[j+1][0][1:-1]])
                    temp_price = rows[j][0][1:-1]
                    temp_price2 = rows[j+1][0][1:-1]

                flag = 0
                break
        # print(i,"--",flag)
        if(flag):
            save_rows.append([unified_rows[i][1], unified_rows[i][0], temp_price,temp_price2])
            print("none"+unified_rows[i][1])
        # save_rows.append(time_rows[i][0],rows[i][0])

    #results存入csv
    with open(save_path, 'w', newline='') as t_file:
       csv_writer = csv.writer(t_file)
       for l in save_rows:
           csv_writer.writerow(l)

if __name__ == '__main__':
    price_path = os.path.join("selfprice.csv")
    time_path = os.path.join("sh.601601_train_data.csv")
    save_path = os.path.join("price_withdate.csv")
    unified_path = os.path.join("unified_results_new.csv")
    num_index = 0
    date_index = 0
    get_unified(price_path, time_path, unified_path, save_path, num_index, date_index)