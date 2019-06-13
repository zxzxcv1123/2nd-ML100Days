import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LinearRegression
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
train=pd.read_csv(path+'titanic_train.csv')
test=pd.read_csv(path+'titanic_test.csv')
train_Y=train['Survived']
ids=test['PassengerId']
train=train.drop(['Survived','PassengerId'],axis=1)
test=test.drop(['PassengerId'],axis=1)
df=pd.concat([train,test])
train_num=train_Y.shape[0]
df.isna().sum()
#特徵工程
label=LabelEncoder()
#sex轉0，1
df['Sex']=label.fit_transform(df['Sex'])
#Age用中位數填補
df['Age']=df['Age'].fillna(df['Age'].median())
#Fare取log去除偏態
df['Fare']=df['Fare'].map(lambda x:np.log(x) if x>0 else 0)
# Title 的 特徵工程 : 將各種頭銜按照類型分類, 最後取 One Hot
df_title=df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
df['Title']=df_title
df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df = pd.get_dummies(df, columns = ["Title"])
# 新建:家庭大小 (Fsize)特徵, 並依照大小分別建獨立欄位
df["Fsize"] = df["SibSp"] + df["Parch"] + 1
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)
#Ticket取前面的字 若為純數值則設為X
ticket=[]
for i in df['Ticket']:
    if not i.isdigit():
        ticket.append(i.replace('.','').replace('/','').strip().split(' ')[0])
    else:
        ticket.append('X')
df['Ticket']=ticket
df=pd.get_dummies(df, columns = ["Ticket"], prefix="T")
#Cabin依照第一碼分類再取One Hot
df['Cabin']=df['Cabin'].apply(lambda x:x[0] if not pd.isnull(x) else 'X' )
df=pd.get_dummies(df,columns=['Cabin'],prefix='Cabin')
# Embarked, Pclass 取 One Hot
df = pd.get_dummies(df, columns = ["Embarked"], prefix="Em")
df["Pclass"] = df["Pclass"].astype("category")
df = pd.get_dummies(df, columns = ["Pclass"], prefix="Pc")
# 捨棄 Name 欄位
df.drop(labels = ["Name"], axis = 1, inplace = True)    
# 將資料最大最小化
df = MinMaxScaler().fit_transform(df)

# 將前述轉換完畢資料 df , 重新切成 train_X, test_X
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

# 使用三種模型 : 邏輯斯迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
lr = LogisticRegression(tol=0.001, penalty='l2', fit_intercept=True, C=1.0)
gdbt = GradientBoostingClassifier(tol=100, subsample=0.75, n_estimators=250, max_features=20,
                                  max_depth=6, learning_rate=0.03)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1, 
                            max_features='sqrt', max_depth=6, bootstrap=True)

#線性回歸預測
lr.fit(train_X,train_Y)
lr_pred=lr.predict_proba(test_X)[:,1]
sub=pd.DataFrame({'PassengerId':ids,'Survived':lr_pred})
sub['Survived']=sub['Survived'].map(lambda x:1 if x>0.5 else 0)
sub.to_csv('titanic.csv',index=False)
# 梯度提升機預測檔 
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': gdbt_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('titanic_gdbt.csv', index=False)
# 隨機森林預測檔
rf.fit(train_X, train_Y)
rf_pred = rf.predict_proba(test_X)[:,1]
sub = pd.DataFrame({'PassengerId': ids, 'Survived': rf_pred})
sub['Survived'] = sub['Survived'].map(lambda x:1 if x>0.5 else 0) 
sub.to_csv('titanic_rf.csv', index=False)
#堆疊泛化
from mlxtend.classifier import StackingClassifier
mete_estimator=GradientBoostingClassifier(tol=100, subsample=0.70, n_estimators=50,max_features='sqrt', max_depth=4, learning_rate=0.3)
stacking=StackingClassifier(classifiers=[lr,gdbt,rf],meta_classifier=mete_estimator)
stacking.fit(train_X,train_Y)
stacking_pred=stacking.predict(test_X)
sub=pd.DataFrame({'PassengerId': ids, 'Survived': stacking_pred})
sub.to_csv('titanic_stacking.csv',index=False)
