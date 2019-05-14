import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'train.csv')
train_y=np.log1p(df_train['SalePrice'])
df=df_train.drop(['SalePrice','Id'],axis=1)
#削減文字型蘭為
num_col=[]
for i in df:
    if df[i].dtype!='object':
        num_col.append(i)
df=df[num_col]
#缺失值填補-1
df.isnull().sum()
df=df.fillna(-1)
#顯示GrLivArea與目標值得散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
df['GrLivArea'].describe()
sns.regplot(x=df['GrLivArea'],y=train_y)
#plt.scatter(df['GrLivArea'],train_y)
MMEcoder=MinMaxScaler()
estimator=LinearRegression()


#1:原始資料做回歸觀察分數
train_X=MMEcoder.fit_transform(df)
cross_val_score(estimator,train_X,train_y,cv=5).mean()#0.846
#2:將GrLivArea限制在800到2500以內,『調整』離群值
test=df.copy()
test['GrLivArea']=df['GrLivArea'].clip(800,2500)
sns.regplot(x=test['GrLivArea'],y=train_y)
train_X=MMEcoder.fit_transform(test)
cross_val_score(estimator,train_X,train_y,cv=5).mean()#0.859
#3:將GrLivArea限制在800到2500以內,『刪除』離群值
keep_indexs = (df['GrLivArea']> 800) & (df['GrLivArea']< 2500)
test=df[keep_indexs]
train_y=train_y[keep_indexs]
sns.regplot(test['GrLivArea'],train_y)
train_X=MMEcoder.fit_transform(test)
cross_val_score(estimator,train_X,train_y,cv=5).mean() #0.876



#顯示1stFlrSF與目標值的散佈圖
df['1stFlrSF'].describe()
sns.regplot(df['1stFlrSF'],train_y)
train_X=MMEcoder.fit_transform(df)
cross_val_score(estimator,train_X,train_y,cv=5).mean()
#分數0.8466

#作業1:將1stFlrSF限制在你覺得適合的範圍內, 調整離群值
test=df.copy()
test['1stFlrSF']=df['1stFlrSF'].clip(300,2500)
sns.regplot(test['1stFlrSF'],train_y)
train_X=MMEcoder.fit_transform(test)
cross_val_score(estimator,train_X,train_y,cv=5).mean()
#score:0.8886

#作業2:將1stFlrSF限制在你覺得適合的範圍內,捨棄離群值
test=df.copy()
test=df[(df['1stFlrSF']>300)&(df['1stFlrSF']<2500)]
train_X=MMEcoder.fit_transform(test)
train_y=train_y[(df['1stFlrSF']>300)&(df['1stFlrSF']<2500)]
sns.regplot(test['1stFlrSF'],train_y)
cross_val_score(estimator,train_X,train_y,cv=5).mean()
#score=0.8935