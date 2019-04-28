import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(path,'application_train.csv')
df=pd.read_csv(f_app)
df['AMT_ANNUITY'].describe()
df['AMT_ANNUITY'].isnull().sum() #有12個空值
five_num=[0,25,50,75,100]
#test為去df['AMT_ANNUITY']去空值
test=df['AMT_ANNUITY'][~df['AMT_ANNUITY'].isnull()] #307499*122 ~布林值正負互換
quantile=[np.percentile(test,q=i) for i in five_num]
print(quantile)
#將最大值改成q99
test[test==test.max()]=np.percentile(test,q=99)
quantile=[np.percentile(test,q=i)for i in five_num]
quantile
test.mode() #找眾數1
test.value_counts() #找眾數2

#作業
#列出 AMT_ANNUITY 的q0~q100
quantile=range(1,101,1)
test=df['AMT_ANNUITY'][~df['AMT_ANNUITY'].isnull()]
q0_q100=[np.percentile(test,q=i)for i in quantile]

# 將 AMT_ANNUITY 中的 NAs 暫時以中位數填補
test=df['AMT_ANNUITY']
test.fillna(q0_q100[49],inplace=True)
test.isnull().sum()

#將 AMT_ANNUITY 的數值標準化至 -1 ~ 1 間
#公式:(((x-min)/(max-min))-0.5)**2
def normalize_value(x):
    max_=max(x)
    min_=min(x)
    x=(((x-min_)/(max_-min_))-0.5)*2
    return x
test=normalize_value(test)
test.describe()
#將 AMT_GOOD_PRICE 的 NAs 以眾數填補
df['AMT_GOODS_PRICE'].isnull().sum()
df['AMT_GOODS_PRICE'].mode()
df['AMT_GOODS_PRICE'].fillna(450000,inplace=True)
