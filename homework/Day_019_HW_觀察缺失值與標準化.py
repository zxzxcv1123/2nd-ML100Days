import pandas as pd
import numpy as np
import copy 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'titanic_train.csv')
df_test=pd.read_csv(path+'titanic_test.csv')
df_test.columns^df_train.columns #^代表不同時包含的元素
#重組資料
train_y=df_train['Survived']
sub=pd.DataFrame(df_test['PassengerId'])
train_x=df_train.drop(['PassengerId','Survived'],axis=1)
test_x=df_test.drop(['PassengerId'],axis=1)
df=pd.concat([train_x,test_x])
#取出int、float的欄位存放於num_features
num_features=[]
for i in df.columns:
    if df[i].dtype=='float64' or df[i].dtype=='int64':
        num_features.append(i)
#削減文字型欄位, 只剩數值型欄位
df=df[num_features]
train_num = train_y.shape[0] #train的筆數
#作業1:測試哪種方式填補缺失值較好
df.isna().sum() #age有263個缺失值 Fare有一個
for i in df:
    df[i].plot.hist() #age左偏
    plt.title(i)
    plt.show()
estimator=LogisticRegression()
#填補-1
df_m1=df.fillna(-1)
train_x=df_m1[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.696
#填補0
df_m2=df.fillna(0)
train_x=df_m2[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.697
#填補中位數
df_m3=df.fillna(df.median())
train_x=df_m3[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.699
#填補平均數
df_m4=df.fillna(df.mean())
train_x=df_m4[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.0.698
#發現因為age是左偏所以用中位數效果較好

#作業2:測試標準化根最大最小化哪個效果好
#最大最小化
df_m3=MinMaxScaler().fit_transform(df_m3)
train_x=df_m3[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.698
#標準化
df_m3=StandardScaler().fit_transform(df_m3)
train_x=df_m3[:train_num]
cross_val_score(estimator,train_x,train_y,cv=5).mean()#0.697
#發現原值效果較好

