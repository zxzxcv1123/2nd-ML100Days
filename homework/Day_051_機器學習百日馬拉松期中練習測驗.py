import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
#匯入資料
path='C:/Users/user/Desktop/JupyterNotebook/exam/'
df=pd.read_csv(path+'train_offline.csv')
test=pd.read_csv(path+'test_offline.csv')
df.isna().sum()
test.isna().sum()
#去除沒優惠券的資料
test=test[~test['Coupon_id'].isna()]
df=df[~df['Coupon_id'].isna()] #刪除沒有優惠卷的資料
sub=test.copy()
#將時間轉為時間格式 並新增時間差的欄位
df['Date']=pd.to_datetime(df['Date'],format='%Y%m%d')
df['Date_received']=pd.to_datetime(df['Date_received'],format='%Y%m%d')
df['Date_delta']=df['Date']-df['Date_received']
sub['Date_received']=pd.to_datetime(sub['Date_received'],format='%Y%m%d')
#創造目標欄位
df['label']=df['Date_delta'].apply(lambda x:1 if x<=pd.Timedelta(15,'D') else 0)
#距離欄位處理
df['Distance'].hist() #先觀察資料分布
df['Distance']=df['Distance'].fillna(df['Distance'].median())
sub['Distance']=sub['Distance'].fillna(sub['Distance'].median())
#連續型資料離散化
#le=LabelEncoder()
#df['equal_width_Distance']=pd.cut(df['Distance'],10)
#df['equal_width_Distance']=le.fit_transform(df['equal_width_Distance'])
#sub['equal_width_Distance']=pd.cut(sub['Distance'],10)
#sub['equal_width_Distance']=le.fit_transform(sub['equal_width_Distance'])
#df['equal_width_Distance'].value_counts().plot.bar()#觀察分布情形
#獲得票券取得星期
df['cupon_week']=df['Date_received'].apply(lambda x:x.weekday())
sub['cupon_week']=sub['Date_received'].apply(lambda x:x.weekday())
#票券種類
#df['CouponType']=df['Discount_rate'].apply(lambda x:1 if ':' in x else 0)
#sub['CouponType']=sub['Discount_rate'].apply(lambda x:1 if ':' in x else 0)
#票券折扣率
def rate(row):
    if ":" in row:
        rows=row.split(':')
        return float(rows[1])/float(rows[0])
    else:
        return float(row)
df['CoupinRate']=df['Discount_rate'].astype('str').apply(rate)
sub['CoupinRate']=sub['Discount_rate'].astype('str').apply(rate)
#滿額多少才能折扣
def get(row):
    if ':' in row:
        rows=row.split(':')
        return int(rows[0])
    else:
        return 0
df['getDiscount']=df['Discount_rate'].astype('str').apply(get)
sub['getDiscount']=sub['Discount_rate'].astype('str').apply(get)
#Merchant_id 轉文字
#df['Merchant_id']=df['Merchant_id'].astype('str')
#sub['Merchant_id']=sub['Merchant_id'].astype('str')
#新增日期
df['Date_received']=df['Date_received'].astype(str).apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
df['Day']=df['Date_received'].apply(lambda x:datetime.strftime(x,'%d')).astype('int64')
sub['Date_received']=sub['Date_received'].astype(str).apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
sub['Day']=sub['Date_received'].apply(lambda x:datetime.strftime(x,'%d')).astype('int64')
#有無購物
df['buy']=df['Date'].apply(lambda x:1 if not pd.isna(x) else 0)
#df.isna().sum()
##找特徵
#df.groupby('buy')['Distance'].value_counts().plot.bar() #target為0的過多
#target1=np.where(df['buy']==1)[0]
#target0=np.where(df['buy']==0)[0]
#target0=target0[:42936]
#target=np.concatenate((target0,target1))
#dff=df.iloc[target,:]
#dff.isna().sum()
#要訓練的特徵
df.columns
feature=['Distance','cupon_week','CoupinRate','getDiscount','Day']
#分割資料
#train_X,test_X,train_Y,test_Y=train_test_split(df[feature],df['label'],test_size=0.25,random_state=42)
def train_valid_split(row,data_cut='20160416'):
    is_train=True if pd.to_datetime(row,format='%Y%m%d')<pd.to_datetime(data_cut,format='%Y%m%d') else False
    return is_train
df['is_train']=df['Date_received'].apply(train_valid_split)
train=df[df['is_train']]
valid=df[~df['is_train']]
train.reset_index(drop=True,inplace=True)
valid.reset_index(drop=True,inplace=True)
#print("Train size: {}, #positive: {}".format(len(train), train["label"].sum()))
#print("Valid size: {}, #positive: {}".format(len(valid), valid["label"].sum()))
#調整參數
#rf=RandomForestClassifier()
#parameter={'max_depth':[3,5,7,9],
#           'min_samples_leaf':[1,3,5,9]}
#search=GridSearchCV(rf,parameter,scoring='accuracy',n_jobs=-1,verbose=1)
#search.fit(df_X,df_Y)
#print(f'Best Accuracy:{search.best_score_}')
#print(f'參數:{search.best_params_}') min_samples_leaf=9,max_depth=9
#隨機森林
#rf=RandomForestClassifier(n_estimators=100,min_samples_leaf=9,max_depth=9)
#rf.fit(df_X,df_Y)
#rf_pred=rf.predict_proba(test_X)
#print(f'AUC:{roc_auc_score(test_Y,rf_pred[:,1])}') #AUC=0.86
#梯度提升機
#gdbt=GradientBoostingClassifier(n_estimators=300,max_depth=5,min_samples_leaf=10,n_jobs=-1)
#gdbt.fit(train[feature],train['label'])
#gdbt_pred=gdbt.predict_proba(valid[feature])
#print('AUC:',roc_auc_score(valid['label'],gdbt_pred[:,1])) #AUC=0.7669
#print(gdbt.feature_importances_)
##LogisticRegression
#logistic=LogisticRegression()
#logistic.fit(train[feature],train['label'])
#log_pred=logistic.predict_proba(valid[feature])[:,1]
#print('AUC_Score:',roc_auc_score(valid['label'],log_pred)) #0.7757
##極限梯度提升 XGBoost
sgbc=XGBClassifier()
sgbc.fit(train[feature],train['label'])
sgbc_pred=sgbc.predict_proba(valid[feature])[:,1]
print('AUC_Score:',roc_auc_score(valid['label'],sgbc_pred))  #0.7624
#預測提交資料
pred=sgbc.predict_proba(sub[feature])[:,1]
sub['label']=pred
subb=pd.concat((test[["User_id", "Coupon_id", "Date_received"]],sub['label']),axis=1)
subb["User_id"] =subb["User_id"].apply(lambda x:str(int(x)))
subb["Coupon_id"] = subb["Coupon_id"].apply(lambda x:str(int(x)))
subb["Date_received"] = subb["Date_received"].apply(lambda x:str(int(x)))
subb['uid']=subb[["User_id", "Coupon_id", "Date_received"]].apply(lambda x:"_".join(x),axis=1)

out=subb.groupby('uid',as_index=False).mean()
out.to_csv('0615XGBoosting.csv',index=False)
