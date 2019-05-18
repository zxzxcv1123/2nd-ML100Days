import pandas as pd 
import numpy as np
import copy,time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'titanic_train.csv')
df_test=pd.read_csv(path+'titanic_test.csv')
train_Y=df_train['Survived']
ids=df_test['PassengerId']
df=pd.concat([df_train,df_test])
df=df.drop(['Survived','PassengerId'],axis=1)
#取出類別型欄位
object_features=[]
for c in df:
    if df[c].dtype=='object':
        object_features.append(c)
df=df[object_features]
df=df.fillna('None')
train_num=train_Y.shape[0]
#標籤編碼+線性回歸
df_temp=pd.DataFrame()
for c in df:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
train_X=df_temp[:train_num]
estimator=LogisticRegression()
score=cross_val_score(estimator,train_X,train_Y,cv=5).mean()
print(score) #0.7800
#均值編碼+線性回歸
data=pd.concat([df[:train_num],train_Y],axis=1)
for c in df.columns:
    mean_df=data.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns=[c,f'{c}_mean']
    data=pd.merge(data,mean_df,on=c,how='left')
    data=data.drop([c],axis=1)
data=data.drop(['Survived'],axis=1)
estimator=LogisticRegression()
score=cross_val_score(estimator,data,train_Y,cv=5).mean()
score#1 
#均值編碼+梯度提升樹
estimator=GradientBoostingRegressor()
cross_val_score(estimator,data,train_Y,cv=5).mean() #0.999
