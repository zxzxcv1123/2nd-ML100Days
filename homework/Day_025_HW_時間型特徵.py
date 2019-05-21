import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df=pd.read_csv(path+'taxi_data1.csv')
train_Y=df['fare_amount']
df=df.drop(['fare_amount'],axis=1)
#使用strptime解析(parse)時間字串 /strftime匯出時間格式(format)
df['pickup_datetime']=df['pickup_datetime'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S UTC'))
df['picked_year']=df['pickup_datetime'].apply(lambda x:datetime.datetime.strftime(x,'%Y')).astype('int64')
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')
#將結果使用線性迴歸 / 梯度提升樹分別看結果
df_temp=df.drop(['pickup_datetime'],axis=1)
scaler=MinMaxScaler()
train_X=scaler.fit_transform(df_temp)
linear=LinearRegression()
cross_val_score(linear,train_X,train_Y,cv=5).mean() #線性回歸0.2687
gradient=GradientBoostingRegressor()
cross_val_score(gradient,train_X,train_Y,cv=5).mean() #梯度提升樹0.7114
#加上星期特徵做線性回歸、梯度上升樹
df['picked_week']=df['pickup_datetime'].apply(lambda x:x.weekday())
df_temp=df.drop(['pickup_datetime'],axis=1)
train_X=scaler.fit_transform(df_temp)
cross_val_score(linear,train_X,train_Y,cv=5).mean() #0.02657
cross_val_score(gradient,train_X,train_Y,cv=5).mean() #0.7109
#加上日週期
import math
df['day_cycle']=df['pickup_hour']/12+df['pickup_minute']/720 + df['pickup_second']/43200
df['day_cycle']=df['day_cycle'].map(lambda x:math.sin(x*math.pi))
df_temp=df.drop(['pickup_datetime'],axis=1)
train_X=scaler.fit_transform(df_temp)
cross_val_score(linear,train_X,train_Y,cv=5).mean() #0.02617
cross_val_score(gradient,train_X,train_Y,cv=5).mean() #0.7089

