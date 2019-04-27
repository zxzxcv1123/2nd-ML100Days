import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path='C:/Users/user/Desktop/JupyterNotebook'
f_app=os.path.join(path,'application_train.csv')
df=pd.read_csv(f_app)
#檢查是否有異常值
#DAYS_BIRTH客戶申請貸款時的年齡 正常
(df['DAYS_BIRTH']/-365).describe() 
#貸款時已工作時間 異常
(df['DAYS_EMPLOYED'] / 365).describe()
sns.boxplot(df['DAYS_EMPLOYED'])
#處理離群值
#1刪除列
test=df['DAYS_EMPLOYED'][df['DAYS_EMPLOYED']!=365243]
#2視為nan
test2=df['DAYS_EMPLOYED']
test2.replace({365243:np.nan},inplace=True)
test2.plot.hist()
#檢查OWN_CAR_AGE:貸款人的車齡
plt.hist(df['OWN_CAR_AGE'])
df['OWN_CAR_AGE'][df['OWN_CAR_AGE']>60].value_counts()
#發現車齡64.65的人特別多

#作業 判斷哪寫可能有離群值 為甚麼
#一: AMT_INCOME_TOTAL 
plt.boxplot(df['AMT_INCOME_TOTAL'])
df['AMT_INCOME_TOTAL'].describe()
sns.boxenplot(df['AMT_INCOME_TOTAL'])
#11700000000明顯為離群值，收入過高常理上也不需貸款

#二:家庭人數20可能為輸入錯誤 
plt.boxplot(df['CNT_FAM_MEMBERS'])
df['CNT_FAM_MEMBERS'].describe()

#三:CNT_CHILDREN 最大值有19個小孩
sns.boxplot(df['CNT_CHILDREN'])
df['CNT_CHILDREN'].describe()





