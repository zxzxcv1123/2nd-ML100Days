import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plot
from sklearn.linear_model import Lasso
from itertools import compress
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df=pd.read_csv(path+'titanic_train.csv')
train_Y=df['Survived']
df=df.drop(['PassengerId'],axis=1)
#計算df整體相關係數, 並繪製成熱圖
corr=df.corr()
sns.heatmap(corr)
#篩選相關系大於0.1或-0.1
high_list=list(corr[(corr['Survived']>0.1)|(corr['Survived']<-0.1)].index)
high_list.pop(0) #將survived移除
#刪除Survived
df=df.drop(['Survived'],axis=1)
#取出數值型欄位
num_features=[]
for i in df:
    if df[i].dtype!='object':
        num_features.append(i)
df=df[num_features]
df=df.fillna(-1)
scaler=MinMaxScaler()

#原始特徵+邏輯斯回歸
train_X=scaler.fit_transform(df)
estimator=LogisticRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7038

#高相關性特徵+邏輯斯回歸
train_X=scaler.fit_transform(df[high_list])
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.6791

#L1 Embedding(嵌入法)
#調整不同的正規化程度，就會自然使得一部分的特徵係數為０，因此刪除的是係數為０的特徵，不須額外指定門檻，但需調整正規化程度
L1_Reg=Lasso(alpha=0.003)
train_X=scaler.fit_transform(df)
L1_Reg.fit(train_X,train_Y)
L1_Reg.coef_
L1_mask=list(L1_Reg.coef_!=0) #第三個欄位False
L1_list=list(compress(list(df),list(L1_mask))) #使用compress篩選出True的欄位
#L1_Embedding特徵+邏輯斯回歸
train_X=scaler.fit_transform(df[L1_list])
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7049
