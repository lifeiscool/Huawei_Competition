import baostock as bs
import pandas as pd
import numpy as np
import csv
import os

#获取股票价格
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
code='sh.601628'
rs = bs.query_history_k_data(code,
                             "date,code,open,close,high,low,volume,amount,turn,pbMRQ,pctChg",
                             start_date='2018-01-01', end_date='2019-05-16' ,frequency="d", adjustflag="3")
print( 'query_history_k_data respond error_code:' +rs.error_code)
print( 'query_history_k_data respond  error_msg:' +rs.error_msg)
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

result = pd.DataFrame(data_list, columns=rs.fields)
# result.drop(['adjustflag'],axis=1,inplace=True)
result.to_csv(str( 'data/'+code +"_train_data.csv"), index=False ,)
# print(result)