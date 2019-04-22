import os
import numpy as np
import pandas as pd

# 設定 data_path
os.getcwd()
print(os.listdir("./"))
data_dir='C:/Users/user/Desktop/JupyterNotebook/'

f_app=os.path.join(data_dir,'application_train.csv')
data=pd.read_csv(f_app)
#資料的 row 數以及 column 數
data.describe()
#列出所有欄位
data.columns
#截取部分資料
data.iloc[0:3] #指定第一到第3-1列
data.loc[0:3] #index數字0,1,2,3列

count=data['TARGET'].value_counts() #觀察個數
print(count)

import matplotlib.pyplot as plt
plt.hist(count)