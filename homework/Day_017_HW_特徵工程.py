import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
data_path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(data_path+'train.csv')
df_test=pd.read_csv(data_path+'test.csv')
#先抽離train_Y與ids
train_Y=np.log1p(df_train['SalePrice'])
ids=df_test['Id']
df_train=df_train.drop(['Id','SalePrice'],axis=1)
df_test=df_test.drop(['Id'],axis=1) 
df=pd.concat([df_train,df_test]) #合併資料
# 特徵工程-簡化版: 全部空值先補-1, 所有類別欄位先做 LabelEncoder, 然後再與數字欄位做 MinMaxScaler
# 這邊使用 LabelEncoder 只是先將類別欄位用統一方式轉成數值以便輸入模型, 當然部分欄位做 One-Hot可能會更好, 只是先使用最簡單版本作為範例
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
#提交結果
sub=pd.DataFrame({'Id':ids,'SalePrice':pred})
sub.to_csv('house_baseline.csv',index=False)






#作業1:下列A~E五個程式區塊中，哪一塊是特徵工程? C
#作業2:對照程式區塊 B 與 C 的結果，
#請問那些欄位屬於"類別型欄位"? (回答欄位英文名稱即可) Pclass、Sex、Ticket、Cabin、Embarked
#作業三:哪個欄位是"目標值"? survive
# 程式區塊 A
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data_path = 'C:\\Users\\asus\\Documents\\GitHub\\2nd_ML100Days_temp\\data\\Part02\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape
# 程式區塊 B
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()
# 程式區塊 C
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()
# 程式區塊 D
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)

# 程式區塊 E
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv('titanic_baseline.csv', index=False)