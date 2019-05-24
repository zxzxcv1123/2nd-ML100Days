import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df=pd.read_csv(path+'train.csv')
train_Y=np.log1p(df['SalePrice'])
df=df.drop(['SalePrice'],axis=1)
#生活面積(GrLivArea) 對 販售條件(SaleCondition) 做群聚編碼
df['SaleCondition']=df['SaleCondition'].fillna('None')
mean_df=df.groupby(['SaleCondition'])['GrLivArea'].mean().reset_index()
mode_df=df.groupby(['SaleCondition'])['GrLivArea'].apply(lambda x:x.mode()[0]).reset_index()
median_df=df.groupby(['SaleCondition'])['GrLivArea'].median().reset_index()
max_df=df.groupby(['SaleCondition'])['GrLivArea'].max().reset_index()
temp = pd.merge(mean_df, mode_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, median_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, max_df, how='left', on=['SaleCondition'])
temp.columns = ['SaleCondition', 'Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max']
df=pd.merge(df,temp,how='left',on=['SaleCondition'])
df=df.drop(['SaleCondition'],axis=1)
#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features=[]
for i in df:
    if df[i].dtype!='object':
        num_features.append(i)
df=df[num_features]
df=df.fillna(-1)
scaler=MinMaxScaler()
#沒有這四個新特徵的 dataframe 稱為 df_minus
df_minus=df.drop(['Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max'] , axis=1)


#原始特徵+線性回歸
train_X=scaler.fit_transform(df_minus)
estimator=LinearRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#0.8465

#新特徵+線性回歸
train_X=scaler.fit_transform(df)
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#0.8487

#原始特徵+梯度提升樹
train_X=scaler.fit_transform(df_minus)
estimator=GradientBoostingRegressor()
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#0.8859

#新特徵+梯度提升樹
train_X=scaler.fit_transform(df)
estimator=GradientBoostingRegressor()
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#0.8868
