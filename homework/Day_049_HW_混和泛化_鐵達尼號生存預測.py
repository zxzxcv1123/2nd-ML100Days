import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
train=pd.read_csv(path+'titanic_train.csv')
test=pd.read_csv(path+'titanic_test.csv')
train_Y=train['Survived']
ids=test['PassengerId']
train=train.drop(['Survived','PassengerId'],axis=1)
test=test.drop(['PassengerId'],axis=1)
df=pd.concat([train,test])
train_num=train_Y.shape[0]
#檢查空缺值
df.isna().sum()

#特徵工程
#Sex : 直接轉男 0 女 1
df['Sex']=df['Sex'].map({'male':0,'female':1})
#Fare:用 log 去偏態, 0 則直接取 0
df['Fare']=df['Fare'].map(lambda x:np.log(x) if x>0 else 0)
#age補中位數
df['Age']=df['Age'].fillna(df['Age'].median())
# Title 的 特徵工程 : 將各種頭銜按照類型分類, 最後取 One Hot
df_title=[i.split(',')[1].split('.')[0].strip() for i in df['Name']]
df['Title']=pd.Series(df_title)
df["Title"]=df["Title"].replace(['Lady','the Countess','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
df['Title']=df['Title'].map({"Master":0,"Miss":1,"Ms":1,"Mme":1,"Mlle":1,"Mrs":1,"Mr":2,"Rare":3})
df['Title']=df['Title'].astype('int')
df=pd.get_dummies(df,columns=['Title']) #獨熱編碼
#新建家庭大小 (Fsize)特徵, 並依照大小分別建獨立欄位
df['Fsize']=df['SibSp']+df['Parch']+1
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)
# Ticket : 如果不只是數字-取第一個空白之前的字串(去除'.'與'/'), 如果只是數字-設為'X', 最後再取 One Hot
ticket=[]
for i in list(df.Ticket):
    if not i.isdigit():#isdigit檢測字符串是否由數字組成
        ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0]) #strip清除首尾的空白
    else:
        ticket.append('X')
df['Ticket']=ticket
df=pd.get_dummies(df,columns=['Ticket'],prefix='T')
# Cabib 依照第一碼分類, 再取 One Hot
df['Cabin']=pd.Series(i[0] if not pd.isnull(i) else 'X' for i in df['Cabin'])
df=pd.get_dummies(df,columns=['Cabin'],prefix='Cabin')
# Embarked, Pclass 取 One Hot
df = pd.get_dummies(df, columns = ["Embarked"], prefix="Em")
df["Pclass"] = df["Pclass"].astype("category")
df = pd.get_dummies(df, columns = ["Pclass"], prefix="Pc")
# 捨棄 Name 欄位
df.drop(labels = ["Name"], axis = 1, inplace = True)

#資料縮放
df=MinMaxScaler().fit_transform(df)
train_X=df[:train_num]
test_X=df[train_num:]

#邏輯斯回歸
lr=LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
lr.fit(train_X, train_Y)
lr_pred = lr.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': lr_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('titanic_lr.csv', index=False) 
#梯度提升機
gdbt=GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=20,max_depth=6, learning_rate=0.03)
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': gdbt_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('titanic_gdbt.csv', index=False)
# 隨機森林預測檔
rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1,max_features='sqrt', max_depth=6, bootstrap=True)
rf.fit(train_X, train_Y)
rf_pred = rf.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': rf_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('titanic_rf.csv', index=False)
# 混合泛化預測檔 
blending_pred = lr_pred*0.33+ gdbt_pred*0.33+ rf_pred*0.34
sub = pd.DataFrame({'PassengerId': ids, 'Survived': blending_pred})
sub['Survived']=sub['Survived'].apply(lambda x:1 if x>0.5 else 0)
sub.to_csv('titanic_blending.csv', index=False)
