import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
data_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(data_dir,'application_train.csv')
data=pd.read_csv(f_app)
#看DAYS_EMPLOYED、AMT_INCOME_TOTAL的關係
sub_df=data[data['DAYS_EMPLOYED']!=365243] #去離群值
plt.scatter(sub_df['DAYS_EMPLOYED']/(-365),sub_df['AMT_INCOME_TOTAL'])
plt.xlabel('DAYS_EMPLOYED')
plt.ylabel('AMT_INCOME_TOTAL')
plt.show()
#發現看不出任何型態，因此對數值範圍較大的取Log
log_ATM=np.log10(sub_df['AMT_INCOME_TOTAL'])
#再次畫圖
plt.scatter(sub_df['DAYS_EMPLOYED']/(-365),log_ATM)
plt.xlabel('DAYS_EMPLOYED')
plt.ylabel('AMT_INCOME_TOTAL')
plt.show()
#觀察相關係數
corr=np.corrcoef(sub_df['DAYS_EMPLOYED']/(-365),log_ATM)

#作業觀察data的相關係數
le=LabelEncoder()
for col in data:
    if data[col].dtype=='object':
        if len(data[col].unique())<3:
            le.fit(data[col])
            data[col]=le.transform(data[col])
cor_target=data.corr()['TARGET']
cor_target=cor_target.sort_values(ascending=False)
cor_target=cor_target.drop('TARGET')       
#列出相關最大的5筆
max_=cor_target[0:15]
#最小的5筆
min_=cor_target[-16:-1]
#大小關係圖
fig=plt.figure(figsize=(90,10))
ax1=fig.add_subplot(1,2,1) #1*2的第一格
ax2=fig.add_subplot(1,2,2)
ax1.bar(max_.index,height=max_)
ax2.bar(min_.index,height=min_)
#查看關係
data.boxplot(column='EXT_SOURCE_3',by='TARGET')

