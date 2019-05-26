import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df=pd.read_csv(path+'titanic_train.csv')
train_Y=df['Survived']
df=df.drop(['PassengerId','Survived'],axis=1)
#簡單的特徵工程
encoder=LabelEncoder()
scaler=MinMaxScaler()
df=df.fillna(-1)
for c in df:
    if df[c].dtype=='object':
        df[c]=encoder.fit_transform(list(df[c]))
    df[c]=scaler.fit_transform(df[c].values.reshape(-1,1))
#隨機森林擬合後, 將結果依照重要性由高到低排序
estimator=RandomForestClassifier()
estimator.fit(df,train_Y)
print(estimator.feature_importances_)
feats=pd.Series(data=estimator.feature_importances_,index=df.columns)
feats=feats.sort_values(ascending=False)
#原始特徵+隨機森林
train_X=scaler.fit_transform(df)
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.8272
#高重要性特徵+隨機森林(前5重要特徵)
high_feature=list(feats[:5].index)
train_X=scaler.fit_transform(df[high_feature])
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7968
#觀察重要特徵與目標的分布
sns.regplot(x=df['Sex'],y=train_Y,fit_reg=False)
sns.regplot(x=df['Fare'],y=train_Y,fit_reg=False)
#製作新特徵看看效果
df['add']=(df['Sex']+df['Fare'])/2
df['multi']=(df['Sex']*df['Fare'])
train_X=scaler.fit_transform(df)
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.8148

