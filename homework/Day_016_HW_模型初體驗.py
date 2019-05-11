import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#匯入檔案
path='C:/Users/user/Desktop/JupyterNotebook/'
f_app_train=os.path.join(path,'application_train.csv')
f_app_test=os.path.join(path,'application_test.csv')
app_train=pd.read_csv(f_app_train)
app_test=pd.read_csv(f_app_test)
#Encoding(物件型態，且資料<=2)
le=LabelEncoder()
le_count=0 #紀錄幾個欄位被轉換
for i in app_train:
    if app_train[i].dtype=='object':
        if len(app_train[i].unique())<3:
            le.fit(app_train[i])
            app_train[i]=le.transform(app_train[i])
            app_test[i]=le.transform(app_test[i])
            le_count+=1
#標籤編碼 (2種類別) 欄位轉 One Hot Encoding
app_train=pd.get_dummies(app_train)
app_test=pd.get_dummies(app_test)
#離群值
app_train['DAYS_EMPLOYED_ANOM']=app_train['DAYS_EMPLOYED']==365243
app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
app_test['DAYS_EMPLOYED_ANOM']=app_test['DAYS_EMPLOYED']==365243
app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
#出生日期轉絕對值
app_train['DAYS_BIRTH']=abs(app_train['DAYS_BIRTH'])
app_test['DAYS_BIRTH']=abs(app_test['DAYS_BIRTH'])
#將欄位出現在test而沒有出現在train的去除
train_labels=app_train['TARGET']
app_train,app_test=app_train.align(app_test,join='inner',axis=1)
#特徵欄位清單
train=app_train
features=list(train.columns)
#複製test資料
test=app_test.copy()
#填補缺失值
imputer=Imputer(strategy='median')
imputer.fit(train)
train=imputer.transform(train)
test=imputer.transform(test)
np.isnan(train).any() #結果False代表沒缺失值
#特徵縮放
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(train)
train=scaler.transform(train)
test=scaler.transform(test)

#fit the model
#設定參數
log_reg=LogisticRegression(C=0.0001) #C默認1越小表示越強正則化
#訓練
log_reg.fit(train,train_labels)
#預測成功的機率(只留下1的機率這排)
log_reg_pred=log_reg.predict_proba(test)[:,1]
#儲存預測結果
submit=app_test[['SK_ID_CURR']]
submit['TARGET']=log_reg_pred

submit.to_csv('sample_submission.csv',index=False)
