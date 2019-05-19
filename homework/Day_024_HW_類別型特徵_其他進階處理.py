import pandas as pd
import numpy as np
import copy,time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'titanic_train.csv')
df_test=pd.read_csv(path+'titanic_test.csv')
train_Y=df_train['Survived']
ids=df_train['PassengerId']
df=pd.concat([df_train,df_test])
df=df.drop(['PassengerId','Survived'],axis=1)
#類別特徵
object_features=[]
for i in df:
    if df[i].dtype=='object':
        object_features.append(i)
df=df[object_features]
df=df.fillna("None")
train_num=train_Y.shape[0]
df.nunique()
#標籤編碼+LogisticRegression
df_temp=pd.DataFrame()
for c in df:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
estimator=LogisticRegression()
train_X=df_temp[:train_num]
cross_val_score(estimator,train_X,train_Y,cv=5).mean()#0.7800


#計數編碼---------------------------------------------------------
#df.groupby(['Ticket']).size(),但欄位名稱會變成size,要取別名就需要用語法 df.groupby(['Ticket']).agg({'Ticket_Count':'size'})
count_df=df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# 但是上面資料表結果只是 'Ticket' 名稱對應的次數, 要做計數編碼還需要第二行:將上表結果與原表格merge,合併於'Ticket'欄位
# 使用 how='left'是完全保留原資料表的所有index與順序
df=pd.merge(df,count_df,on=['Ticket'],how="left")
count_df.sort_values(by=['Ticket_Count'],ascending=False)
#'Ticket'計數編碼 + 邏輯斯迴歸
df_temp=pd.DataFrame()
for c in df:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
train_X=df_temp[:train_num]
estimator=LogisticRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7799
#特徵雜湊---------------------------------------------------------
#Ticket特徵雜湊+羅吉斯回歸
df_temp = pd.DataFrame()
for c in object_features:
    df_temp[c] = LabelEncoder().fit_transform(df[c])
#這邊的雜湊編碼,是直接將'Ticket'的名稱放入雜湊函數的輸出數值,為了要確定是緊密(dense)特徵,因此除以10後看餘數
#這邊的10是隨機選擇,不一定要用10,同學可以自由選擇購小的數字試看看.基本上效果都不會太好
df_temp['Ticket_Hash']=df['Ticket'].map(lambda x:hash(x)%10)
train_X=df_temp[:train_num]
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7777
#------------------------------------------------------------------------------
#'Ticket'計數編碼 + 'Ticket'特徵雜湊 + 邏輯斯迴歸
df_temp=pd.DataFrame()
for c in df:
    df_temp[c]=LabelEncoder().fit_transform(df[c])
#雜湊
df_temp['Ticket_Hash']=df_temp['Ticket'].map(lambda x:hash(x)%10)
#計數
count_df=df_temp.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
df_temp=pd.merge(df_temp,count_df,on=['Ticket'],how='left')
train_X=df_temp[:train_num]
#算分
estimator=LogisticRegression()
cross_val_score(estimator,train_X,train_Y,cv=5).mean() #0.7744
