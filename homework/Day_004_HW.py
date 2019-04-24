import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#匯入資料
path='C:/Users/user/Desktop/JupyterNotebook'
train_app=os.path.join(path,'application_train.csv')
test_app=os.path.join(path,'application_test.csv')
train=pd.read_csv(train_app)
test=pd.read_csv(test_app)
#t查看欄位數
train.shape
#查看欄位
train.columns
#查看各資料類型各有多少欄位
train.dtypes.value_counts()
#查看空值
train.isnull().sum()
#檢視資料中各自類別的數量
train.select_dtypes(include=['object']).apply(pd.Series.nunique,axis=0)
#視覺化
train['TARGET'].plot.hist()
#Lebel encoding(適用於資料有大小關係，)
from sklearn.preprocessing import LabelEncoder
labelencoding=LabelEncoder()
le_count=0
for col in train:
    if train[col].dtype=='object':
        if len(train[col].unique())<=2: #取數量<=2的
            labelencoding.fit(train[col])
            train[col]=labelencoding.transform(train[col])
            test[col]=labelencoding.transform(test[col])
            le_count+=1
print('%d columns were label ecoded'% le_count)
#one hot encoding
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#作業
sub_train = pd.DataFrame(train['WEEKDAY_APPR_PROCESS_START'])
sub_train.shape #1欄
sub_train.head()
#one hot encoding
sub_train=pd.get_dummies(sub_train)
sub_train.shape #7欄
sub_train.head()
