import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from scipy import stats
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'titanic_train.csv')
df_test=pd.read_csv(path+'titanic_test.csv')
#資料重組
train_Y=df_train['Survived']
ids=df_test['PassengerId']
df_train=df_train.drop(['PassengerId','Survived'],axis=1)
df_test=df_test.drop(['PassengerId'],axis=1)
df=pd.concat([df_train,df_test])
#取術數值型欄位
df.dtypes.value_counts()
num_feature=[]
for i in df:
    if df[i].dtype!='object':
        num_feature.append(i)
#削減文字欄位
df=df[num_feature]
df=df.fillna(0)
train_num=train_Y.shape[0]

#計算基礎分數
encoding=MinMaxScaler()
estimator=LogisticRegression()
df_mm=encoding.fit_transform(df)
train_X=df_mm[:train_num]
cross_val_score(estimator,train_X,train_Y,cv=5).mean()
#score:0.7016
#顯示Fare的分布
sns.distplot(df['Fare'][:train_num])

#作業1:對Fare使用log1p，看結果是否更好
train_X=df.copy()
train_X['Fare']=np.log1p(train_X['Fare'])
sns.distplot(train_X['Fare']) #觀察分布
train_X=encoding.fit_transform(train_X)
train_X=train_X[:train_num]
cross_val_score(estimator,train_X,train_Y,cv=5).mean()
#score:0.7106

#作業二:對Fare取boxcox後，觀察分布，並計算分數
train_X=df.copy()
train_X['Fare'].replace({0:1},inplace=True)
train_X['Fare']=stats.boxcox(train_X['Fare'],lmbda=0.15)
sns.distplot(train_X['Fare'])
train_X=encoding.fit_transform(train_X)
train_X=train_X[:train_num]
cross_val_score(estimator,train_X,train_Y,cv=5).mean()
#score:0.7095
