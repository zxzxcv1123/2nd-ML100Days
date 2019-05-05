import os
import numpy as np
import pandas as pd

# 設定 df_path
#os.getcwd()
#print(os.listdir("./"))
df_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(df_dir,'application_train.csv')
df=pd.read_csv(f_app)

#資料的 row 數以及 column 數
df.describe()
#列出所有欄位
df.columns
#截取部分資料
df.iloc[0:3] #指定第一到第3-1列
df.loc[0:3] #index數字0,1,2,3列

count=df['TARGET'].value_counts() #觀察個數
print(count)

import matplotlib.pyplot as plt
plt.hist(count)