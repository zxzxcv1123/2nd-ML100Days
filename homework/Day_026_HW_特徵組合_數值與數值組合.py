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
# 時間特徵分解方式:使用datetime
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')
df['pickup_month'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%m')).astype('int64')
df['pickup_day'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%d')).astype('int64')
df['pickup_hour'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%H')).astype('int64')
df['pickup_minute'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%M')).astype('int64')
df['pickup_second'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%S')).astype('int64')
# 將結果使用線性迴歸 / 梯度提升樹分別看結果
df = df.drop(['pickup_datetime'] , axis=1)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(df)
Linear = LinearRegression()
cross_val_score(Linear, train_X, train_Y, cv=5).mean()#0.0268
GDBT = GradientBoostingRegressor()
cross_val_score(GDBT, train_X, train_Y, cv=5).mean() #0.0.7114
# 增加緯度差, 經度差兩個特徵
df['longitude_diff'] = df['dropoff_longitude'] - df['pickup_longitude']
df['latitude_diff'] = df['dropoff_latitude'] - df['pickup_latitude']
# 結果 : 光是用經緯度差, 準確度就有巨幅上升
train_X = scaler.fit_transform(df)
cross_val_score(Linear, train_X, train_Y, cv=5).mean()#0.0269
cross_val_score(GDBT, train_X, train_Y, cv=5).mean()#0.7989
# 增加座標距離特徵
df['distance_2D'] = (df['longitude_diff']**2 + df['latitude_diff']**2)**0.5
df[['distance_2D', 'longitude_diff', 'latitude_diff']].head()
#結果 : 加上座標距離後, 準確度再度上升(包含線性迴歸)
train_X = scaler.fit_transform(df)
cross_val_score(Linear, train_X, train_Y, cv=5).mean()#0.0274
cross_val_score(GDBT, train_X, train_Y, cv=5).mean()#0.8058
#加上經緯度一圈的長度比(經度:緯度=0.75756:1)
import math
df['distance_real']=(df['longitude_diff']**2+(df['latitude_diff']*0.75756)**2)**0.5
train_X=MinMaxScaler().fit_transform(df)
cross_val_score(Linear, train_X, train_Y, cv=5).mean()#0.0302
cross_val_score(GDBT, train_X, train_Y, cv=5).mean()#0.8028
#試著只使用新特徵估計目標值(忽略原特徵)
train_X = scaler.fit_transform(df[['distance_real']])
cross_val_score(Linear, train_X, train_Y, cv=5).mean()#0.0014
cross_val_score(GDBT, train_X, train_Y, cv=5).mean()#0.7360
