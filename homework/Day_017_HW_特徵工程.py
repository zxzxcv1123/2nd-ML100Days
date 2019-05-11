# 程式區塊 A
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
data_path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(data_path+'train.csv')
df_test=pd.read_csv(data_path+'test.csv')
# 程式區塊 B
#先抽離train_Y與ids
train_Y=np.log1p(df_train['SalePrice'])
ids=df_test['Id']
df_train=df_train.drop(['Id','SalePrice'],axis=1)
df_test=df_test.drop(['Id'],axis=1) 
df=pd.concat([df_train,df_test]) #合併資料
# 特徵工程-簡化版: 全部空值先補-1, 所有類別欄位先做 LabelEncoder, 然後再與數字欄位做 MinMaxScaler
# 這邊使用 LabelEncoder 只是先將類別欄位用統一方式轉成數值以便輸入模型, 當然部分欄位做 One-Hot可能會更好, 只是先使用最簡單版本作為範例
# 程式區塊 C
le=LabelEncoder()
for c in df:
    if df[c].dtype=='object':
        df[c]=df[c].fillna('None') #填補None
        df[c]=le.fit_transform(df[c]) #encoding
    else: #數值類型
        df[c]=df[c].fillna(-1) #數值型填補負1
#除上述之外,還要把標籤編碼與數值欄位一起做最大最小化,這麼做雖然有些暴力,卻可以最簡單的平衡特徵間影響力
scaler=MinMaxScaler()
df=scaler.fit_transform(df)
# 程式區塊 D
#切割train_X,test_X
train_num=train_Y.shape[0] #1460
train_X=df[:train_num] #前1460筆
test_X=df[train_num:] #剩餘資料
#使用線性回歸模型:train_X,train_Y訓練 並預測test_X
from sklearn.linear_model import LinearRegression
estimator=LinearRegression()
estimator.fit(train_X,train_Y)
pred=estimator.predict(test_X)
pred=np.expm1(pred) #原本數字有取log，因此需還原
# 程式區塊 E
#提交結果
sub=pd.DataFrame({'Id':ids,'SalePrice':pred})
sub.to_csv('house_baseline.csv',index=False)


#作業1:上列A~E五個程式區塊中，哪一塊是特徵工程? C
#作業2:對照程式區塊 B 與 C 的結果，
#請問那些欄位屬於"類別型欄位"? (回答欄位英文名稱即可)
for c in df_train:
    if df_train[c].dtype=='object':
        print(c)
#作業三:哪個欄位是"目標值"? SalePrice