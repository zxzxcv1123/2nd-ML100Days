import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'titanic_train.csv')
df_test=pd.read_csv(path+'titanic_test.csv')
#資料重組
train_Y=df_train['Survived']
ids=df_test['PassengerId']
df=pd.concat([df_train,df_test])
df=df.drop(['Survived','PassengerId'],axis=1)
#取出類別型欄位
object_features=[]
for i in df.columns:
    if df[i].dtype=='object':
        object_features.append(i)
df=df[object_features]
df=df.fillna('None')
train_num=train_Y.shape[0]
#標籤編碼+線性回歸＃0.3349
df_temp=pd.DataFrame()
for c in df.columns:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
train_X=df_temp[:train_num]
estimator=LinearRegression()
start=time.time()
print(f'shape:{train_X.shape}')
print(f'score:{cross_val_score(estimator,train_X,train_Y,cv=5).mean()}')
print(f'time : {time.time() - start} sec')
#獨熱編碼+線性回歸 #0.4165
df_temp=pd.get_dummies(df)
train_X=df_temp[:train_num]
estimator=LinearRegression()
start=time.time()
print(f'shape:{train_X.shape}')
print(f'score:{cross_val_score(estimator,train_X,train_Y,cv=5).mean()}')
print(f'time:{time.time()-start}sec')
#標籤編碼+梯度提升樹 #0.3770
df_temp=pd.DataFrame()
for c in df:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
train_X=df_temp[:train_num]
estimator=GradientBoostingRegressor()
start=time.time()
print(f'shape:{train_X.shape}')
print(f'score:{cross_val_score(estimator,train_X,train_Y,cv=5).mean()}')
print(f'time:{time.time()-start}')
#獨熱編碼+梯度提升樹 #0.3778
df_temp=pd.get_dummies(df)
train_X=df_temp[:train_num]
estimator=GradientBoostingRegressor()
start=time.time()
print(f'shape:{train_X.shape}')
print(f'score:{cross_val_score(estimator,train_X,train_Y,cv=5).mean()}')
print(f'time:{time.time()-start}')

#標籤編碼適合樹狀結構，獨熱編碼適用非樹狀模型
