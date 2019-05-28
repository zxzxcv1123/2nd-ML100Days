import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# 因為擬合(fit)與編碼(transform)需要分開,因此不使用.get_dummy，而採用sklearn的OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df=pd.read_csv(path+'titanic_train.csv')
train_Y=df['Survived']
df=df.drop(['PassengerId','Survived'],axis=1)
#使用最精簡的特徵工程
LEncoder=LabelEncoder()
MMEncoder=MinMaxScaler()
for c in df:
    df[c]=df[c].fillna(-1)
    if df[c].dtype=='object':
        df[c]=LEncoder.fit_transform(list(df[c]))
    df[c]=MMEncoder.fit_transform(df[c].values.reshape(-1,1))
#將訓練及切成三部分train/val/test,採用test驗證而非k-fold交叉驗證
#train用來訓練梯度提升樹,val用來訓練邏輯斯迴歸,test驗證效果
train_X,test_X,train_Y,test_Y=train_test_split(df,train_Y,test_size=0.5)
train_X,val_X,train_Y,val_Y=train_test_split(train_X,train_Y,test_size=0.5)
#梯度提升樹調整參數並擬合後,再將葉編碼(*.apply)結果做獨熱/邏輯斯迴歸
rf=RandomForestClassifier(n_estimators=20,min_samples_split=10,min_samples_leaf=5,max_features=4,max_depth=3,bootstrap=True)
onehot=OneHotEncoder()
lr=LogisticRegression(solver='lbfgs',max_iter=1000)
rf.fit(train_X,train_Y)
rf.apply(train_X).shape #每個樣本有20個estimator,出現最多次的即為估計值,(222,20)[n_samples, n_estimators]
onehot.fit(rf.apply(train_X))
lr.fit(onehot.fit_transform(rf.apply(val_X)),val_Y)
#將隨機森林+葉編碼+邏輯斯回歸結果輸出
pred_rf_lr=lr.predict_proba(onehot.transform(rf.apply(test_X)))[:, 1]
fpr_rf_lr,tpr_rf_lr,_= roc_curve(test_Y, pred_rf_lr)
# 將隨機森林結果輸出
pred_rf=rf.predict_proba(test_X)[:, 1] #取1的機率 
fpr_rf,tpr_rf,_=roc_curve(test_Y,pred_rf)
#將結果繪圖
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Randomforest')
plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()