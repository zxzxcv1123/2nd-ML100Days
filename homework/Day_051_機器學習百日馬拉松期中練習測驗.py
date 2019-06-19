import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from mlxtend.regressor import StackingRegressor
import sklearn.metrics
import seaborn as sns
#匯入資料
path='C:/Users/user/Desktop/JupyterNotebook/exam/'
train_df=pd.read_csv(path+'train_offline.csv')
test_df=pd.read_csv(path+'test_offline.csv')
test_id=test_df[['User_id','Coupon_id','Date_received']]
test_id=test_id.loc[~test_id['Coupon_id'].isna()].reset_index(drop=True)
#Deal with missing values
train_df.isna().sum()
test_df.isna().sum()
total_df=pd.concat([train_df,test_df],axis=0)
#Distance
DistanceFilling_UM=total_df.groupby(['User_id','Merchant_id'])['Distance'].mean().reset_index()
DistanceFilling_UM.columns=['User_id','Merchant_id','DistanceFilling_UM']
DistanceFilling_U=total_df.groupby(['User_id'])['Distance'].mean().reset_index()
DistanceFilling_U.columns=['User_id','DistanceFilling_U']
DistanceFilling_M = total_df.groupby(['Merchant_id'])['Distance'].mean().reset_index()
DistanceFilling_M.columns = ['Merchant_id','DistanceFilling_M']

total_df=pd.merge(total_df,DistanceFilling_UM,on=['User_id','Merchant_id'],how='left')
total_df = pd.merge(total_df,DistanceFilling_U,on = ['User_id'], how = 'left')
total_df = pd.merge(total_df,DistanceFilling_M,on = ['Merchant_id'], how = 'left')


def DistanceMissingFill(data):
    if np.isnan(data['Distance']):
        if not np.isnan(data['DistanceFilling_UM']):
            return int(data['DistanceFilling_UM'])
        elif not np.isnan(data['DistanceFilling_U']):
            return int(data['DistanceFilling_U'])
        elif not  np.isnan(data['DistanceFilling_M']):
            return int(data['DistanceFilling_M'])
    return data['Distance']
total_df['Distance']=total_df.apply(DistanceMissingFill,axis=1)
total_df['Distance'].isna().sum()
total_df.drop(['DistanceFilling_UM','DistanceFilling_U','DistanceFilling_M'],axis=1,inplace=True)
train_df=total_df[:len(train_df)]
test_df=total_df[len(train_df):]
#去除沒優惠券的資料
test_df=test_df[~test_df['Coupon_id'].isna()]
train_df=train_df[~train_df['Coupon_id'].isna()]
#Dont forget to drop date column in testing data
test_df = test_df.drop(['Date'], axis = 1)
#create label
def fifteenDaysChecking(data):
    if not pd.isna(data['Date']): #假設有消費
        time_diff=pd.to_datetime(data['Date'],format='%Y%m%d')-pd.to_datetime(data['Date_received'],format='%Y%m%d')
        if time_diff<=pd.Timedelta(15,'D'):
            return 1
    return 0
train_df['label']=train_df.apply(fifteenDaysChecking,axis=1)
train_df=train_df.drop(['Date'],axis=1)
#Merge train &test dataset for processing
train_label=train_df['label']
train_df = train_df.drop(['label'], axis = 1)
total_df = pd.concat([train_df,test_df], axis = 0)
#Feature Engineering
#check the unique counts of each features
total_df.nunique()
#convert Date_received to str type for datetime processing
total_df['Date_received']=total_df['Date_received'].astype('int').astype('str')
#convert Date_received to other time information
total_df['Date_received']=total_df['Date_received'].apply(lambda x:datetime.datetime.strptime(x,"%Y%m%d"))
total_df['Month_received']=total_df['Date_received'].apply(lambda x:datetime.datetime.strftime(x,'%m')).astype('int64')
total_df['Day_received']=total_df['Date_received'].apply(lambda x : datetime.datetime.strftime(x,"%d")).astype("int64")
total_df['Month_Cycle']=total_df['Day_received'].map(lambda x:1 if x<=15 else 0)
#Record the total days to June
def set2June(data):
    if data['Month_received']<6: #6月前收到的計算多少天到6月
        return (6-data['Month_received'])*30 - data['Day_received']
    return 1 #6月份收到優惠券的return 1
total_df['CloseToJune']=total_df.apply(set2June,axis=1)
#Coupon
#CouponType
total_df['DiscountType']=total_df.Discount_rate.map(lambda x:1 if (':' in x) else 0)
#what price need to reach that allow to get discount 
total_df['DiscountBound']=total_df.Discount_rate.map(lambda x:int(x.split(':')[0]) if (':' in x) else 0)
#when the price reach to discount, how much money can be discount 
total_df['DirectPriceCut']=total_df.Discount_rate.map(lambda x:int(x.split(':')[1]) if (':' in x) else 0)
#how the discount ratio of both discount type 
total_df['DiscountRatio']=total_df.Discount_rate.map(lambda x:(1-float(x.split(':')[1])/float(x.split(':')[0])) if(':' in x) else float(x))
#how much money we need to cost after reach the distcount bound
total_df['MoneyCost']=total_df['DiscountBound']-total_df['DirectPriceCut']
total_df=total_df.drop(['Discount_rate'],axis=1)
#Check the data distribution fitst
total_df['DiscountBound'].hist()
plt.title('Histogram of Discount Bound')
plt.xlabel('DiscountBound')
plt.ylabel('Count')
plt.show()
#group the Discountbound
cuttingArr=[-1,80,150,250,total_df['DiscountBound'].max()+1]
total_df['DiscountBound_Group'] = pd.cut(total_df.DiscountBound,cuttingArr)
#encoding intervals into integer
le=LabelEncoder()
total_df['DiscountBound_Group']=le.fit_transform(total_df['DiscountBound_Group'])
#find some characteristic or behavior of user
Same_Merchant_User_received=total_df[['User_id','Merchant_id']]
Same_Merchant_User_received['temp']=1
Same_Merchant_User_received=Same_Merchant_User_received.groupby(['User_id','Merchant_id']).agg('sum').reset_index()
Same_Merchant_User_received.columns = ['User_id','Merchant_id','Same_Merchant_User_received']
total_df = pd.merge(total_df, Same_Merchant_User_received, on = ['User_id','Merchant_id'], how = 'left')
#Record how many coupons each user received
Total_Coupon_User_received = total_df.groupby(['User_id'])['Coupon_id'].count().reset_index()
Total_Coupon_User_received.columns = ['User_id', 'Total_Coupon_User_received']
total_df = pd.merge(total_df,Total_Coupon_User_received, on = ['User_id'], how = 'left')
#how many same coupons each user received
Same_Coupon_User_received = total_df[['User_id', 'Coupon_id']]
Same_Coupon_User_received['temp'] = 1
Same_Coupon_User_received = Same_Coupon_User_received.groupby(['User_id','Coupon_id']).agg('sum').reset_index()
Same_Coupon_User_received.columns = ['User_id','Coupon_id','Same_Coupon_User_received']
total_df = pd.merge(total_df,Same_Coupon_User_received , on = ['User_id', 'Coupon_id'], how = 'left')
# how many same coupons each user received at the same day
Same_Day_Same_Coupon_User_received = total_df[['User_id','Coupon_id','Date_received']]
Same_Day_Same_Coupon_User_received['temp'] = 1
Same_Day_Same_Coupon_User_received = Same_Day_Same_Coupon_User_received.groupby(['User_id','Coupon_id','Date_received']).agg('sum').reset_index()
Same_Day_Same_Coupon_User_received.columns = ['User_id','Coupon_id','Date_received','Same_Day_Same_Coupon_User_received']
total_df = pd.merge(total_df, Same_Day_Same_Coupon_User_received , on = ['User_id','Coupon_id','Date_received'],how='left')
#how many coupons each user received at the same day
Same_Day_Total_Coupon_User_received = total_df[['User_id','Date_received']]
Same_Day_Total_Coupon_User_received['temp'] = 1
Same_Day_Total_Coupon_User_received = Same_Day_Total_Coupon_User_received.groupby(['User_id','Date_received']).agg('sum').reset_index()
Same_Day_Total_Coupon_User_received.columns = ['User_id','Date_received', 'Same_Day_Total_Coupon_User_received']
total_df = pd.merge(total_df, Same_Day_Total_Coupon_User_received, on = ['User_id', 'Date_received'], how ='left')
#the mean/max/min discount upper bound of coupon that each user received
User_Received_Coupon_MoneyCost_Mean = total_df.groupby(['User_id'])['MoneyCost'].mean().reset_index()
User_Received_Coupon_MoneyCost_Mean.columns = ['User_id','User_Received_Coupon_MoneyCost_Mean']

User_Received_Coupon_MoneyCost_Max = total_df.groupby(['User_id'])['MoneyCost'].max().reset_index()
User_Received_Coupon_MoneyCost_Max.columns = ['User_id','User_Received_Coupon_MoneyCost_Max']

User_Received_Coupon_MoneyCost_Min = total_df.groupby(['User_id'])['MoneyCost'].min().reset_index()
User_Received_Coupon_MoneyCost_Min.columns = ['User_id','User_Received_Coupon_MoneyCost_Min']

total_df = pd.merge(total_df,User_Received_Coupon_MoneyCost_Mean, on = ['User_id'], how = 'left')
total_df = pd.merge(total_df,User_Received_Coupon_MoneyCost_Max, on = ['User_id'], how = 'left')
total_df = pd.merge(total_df,User_Received_Coupon_MoneyCost_Min, on = ['User_id'], how = 'left')
#the mean discount ratio of coupon that each user received
User_Received_Coupon_DiscountRatio_Mean = total_df.groupby(['User_id'])['DiscountRatio'].mean().reset_index()
User_Received_Coupon_DiscountRatio_Mean.columns = ['User_id','User_Received_Coupon_DiscountRatio_Mean']
total_df = pd.merge(total_df,User_Received_Coupon_DiscountRatio_Mean, on = ['User_id'], how = 'left')
# the minimal days that user received the same coupon. If the user only got the coupon once, then make the day gaps big.
#This will take a long~ time
def CalMinReceivedGap(data):

    dates = data.split(':')
    if len(dates) == 1:
        return -1 #Means only received once
    MinGap = 365
    for i in range(0,len(dates)-1):
        cur_gap = pd.to_datetime( dates[i] , format = "%Y-%m-%d") - pd.to_datetime(dates[i+1] , format = "%Y-%m-%d")
        cur_gap = abs(cur_gap.days)
        if cur_gap < MinGap:
                MinGap = cur_gap
    return MinGap

Min_Coupon_received_gap = total_df[['User_id','Coupon_id','Date_received']]
Min_Coupon_received_gap.Date_received = Min_Coupon_received_gap.Date_received.astype('str')
Min_Coupon_received_gap =  Min_Coupon_received_gap.groupby(['User_id','Coupon_id'])['Date_received'].agg(lambda x : ':'.join(x)).reset_index()
Min_Coupon_received_gap['Min_Coupon_received_gap'] = Min_Coupon_received_gap.Date_received.apply(CalMinReceivedGap)
Min_Coupon_received_gap = Min_Coupon_received_gap.drop(['Date_received'], axis = 1)

total_df = pd.merge(total_df,Min_Coupon_received_gap,on = ['User_id','Coupon_id'], how ='left')
max_received_gap = total_df.Min_Coupon_received_gap.max()
total_df.Min_Coupon_received_gap = total_df.Min_Coupon_received_gap.map(lambda x : x if x != -1 else max_received_gap)
#he maximum days that user received the same coupon. If the user only got the coupon once, then make the day gaps big
def CalMaxReceivedGap(data):
    dates = data.split(':')
    if len(dates) == 1:
        return -1 #Means only received once
    MaxGap = 0
    for i in range(0,len(dates)-1):
        cur_gap = pd.to_datetime( dates[i] , format = "%Y-%m-%d") - pd.to_datetime(dates[i+1] , format = "%Y-%m-%d")
        cur_gap = abs(cur_gap.days)
        if cur_gap > MaxGap:
                MaxGap = cur_gap
    return MaxGap

Max_Coupon_received_gap = total_df[['User_id','Coupon_id','Date_received']]
Max_Coupon_received_gap.Date_received = Max_Coupon_received_gap.Date_received.astype('str')
Max_Coupon_received_gap =  Max_Coupon_received_gap.groupby(['User_id','Coupon_id'])['Date_received'].agg(lambda x : ':'.join(x)).reset_index()
Max_Coupon_received_gap['Max_Coupon_received_gap'] = Max_Coupon_received_gap.Date_received.apply(CalMaxReceivedGap)
Max_Coupon_received_gap = Max_Coupon_received_gap.drop(['Date_received'], axis = 1)

total_df = pd.merge(total_df,Max_Coupon_received_gap,on = ['User_id','Coupon_id'], how ='left')
max_received_gap = total_df.Max_Coupon_received_gap.max()
total_df.Max_Coupon_received_gap = total_df.Max_Coupon_received_gap.map(lambda x : x if x != -1 else max_received_gap)
#Merchant features
Merchant_DiscountBound_mean = total_df.groupby(['Merchant_id'])['DiscountBound'].mean().reset_index()
Merchant_DiscountBound_mean.columns = ['Merchant_id','Merchant_DiscountBound_mean']

Merchant_DiscountBound_max = total_df.groupby(['Merchant_id'])['DiscountBound'].max().reset_index()
Merchant_DiscountBound_max.columns = ['Merchant_id','Merchant_DiscountBound_max']

Merchant_DiscountBound_min = total_df.groupby(['Merchant_id'])['DiscountBound'].min().reset_index()
Merchant_DiscountBound_min.columns = ['Merchant_id','Merchant_DiscountBound_min']

Merchant_DiscountRatio_max = total_df.groupby(['Merchant_id'])['DiscountRatio'].max().reset_index()
Merchant_DiscountRatio_max.columns = ['Merchant_id','Merchant_DiscountRatio_max']

Merchant_DiscountRatio_min = total_df.groupby(['Merchant_id'])['DiscountRatio'].min().reset_index()
Merchant_DiscountRatio_min.columns = ['Merchant_id','Merchant_DiscountRatio_min']

Merchant_DirectPriceCut_max = total_df.groupby(['Merchant_id'])['DirectPriceCut'].max().reset_index()
Merchant_DirectPriceCut_max.columns = ['Merchant_id','Merchant_DirectPriceCut_max']

Merchant_DirectPriceCut_min = total_df.groupby(['Merchant_id'])['DirectPriceCut'].min().reset_index()
Merchant_DirectPriceCut_min.columns = ['Merchant_id','Merchant_DirectPriceCut_min']

Merchant_DirectPriceCut_mean = total_df.groupby(['Merchant_id'])['DirectPriceCut'].mean().reset_index()
Merchant_DirectPriceCut_mean.columns = ['Merchant_id', 'Merchant_DirectPriceCut_mean']

Merchant_MoneyCost_mean = total_df.groupby(['Merchant_id'])['MoneyCost'].mean().reset_index()
Merchant_MoneyCost_mean.columns = ['Merchant_id', 'Merchant_MoneyCost_mean']

Merchant_MoneyCost_max = total_df.groupby(['Merchant_id'])['MoneyCost'].max().reset_index()
Merchant_MoneyCost_max.columns = ['Merchant_id', 'Merchant_MoneyCost_max']

Merchant_MoneyCost_min = total_df.groupby(['Merchant_id'])['MoneyCost'].min().reset_index()
Merchant_MoneyCost_min.columns = ['Merchant_id', 'Merchant_MoneyCost_min']

total_df = pd.merge(total_df,Merchant_DiscountBound_mean,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DirectPriceCut_mean,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_MoneyCost_mean,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DiscountBound_max, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DiscountBound_min, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DiscountRatio_max, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DiscountRatio_min, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DirectPriceCut_max, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_DirectPriceCut_min, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_MoneyCost_max, on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_MoneyCost_min, on = ['Merchant_id'], how = 'left')
Mcount = total_df.Merchant_id.value_counts().reset_index()
Mcount.columns = ['Merchant_id','Merchant_count']
Ccount = total_df.Coupon_id.value_counts().reset_index()
Ccount.columns = ['Coupon_id','Coupon_count']

total_df = pd.merge(total_df,Mcount,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Ccount,on = ['Coupon_id'], how = 'left')
#Check the correlation again before fill the missing values of Distance
correlation = copy.deepcopy(total_df[:len(train_df)])
correlation = pd.concat([correlation, pd.DataFrame(train_label.values, columns = ['label'], index = correlation.index)], axis = 1)
corr = correlation.corr()
plt.figure(figsize = (30,30))
sns.heatmap(corr, cmap = plt.cm.summer, annot = True) #annot=True,write the data value in each cell
plt.title('Correlation of training dataset')
plt.show()
#slightly chance that these users have similar event location
total_df['Distance'] = total_df.groupby(['CloseToJune'])['Distance'].transform(lambda x : x.fillna(x.mode()[0]))
User_Activity_Distance_Mean = total_df.groupby(['User_id'])['Distance'].mean().reset_index()
User_Activity_Distance_Mean.columns = ['User_id','User_Activity_Distance_Mean']
total_df = pd.merge(total_df,User_Activity_Distance_Mean,on = ['User_id'], how = 'left')

User_Activity_Distance_Max = total_df.groupby(['User_id'])['Distance'].max().reset_index()
User_Activity_Distance_Max.columns = ['User_id','User_Activity_Distance_Max']
total_df = pd.merge(total_df,User_Activity_Distance_Max,on = ['User_id'], how = 'left')

User_Activity_Distance_Min = total_df.groupby(['User_id'])['Distance'].min().reset_index()
User_Activity_Distance_Min.columns = ['User_id','User_Activity_Distance_Min']
total_df = pd.merge(total_df,User_Activity_Distance_Min,on = ['User_id'], how = 'left')
total_df['Bound_Distance'] = total_df['DiscountBound']+1 * (total_df['Distance']+1)
total_df['DiscountRatio_Distance'] = total_df['DiscountRatio'] * (total_df['Distance']+1)
total_df['MoneyCost_Distance'] = total_df['MoneyCost']+1 * (total_df['Distance']+1)
total_df['DirectPriceCut_Distance'] = total_df['DirectPriceCut']+1 / (total_df['Distance']+1) 
Merchant_Distance_mean = total_df.groupby(['Merchant_id'])['Distance'].mean().reset_index()
Merchant_Distance_mean.columns = ['Merchant_id','Merchant_Distance_mean']

Merchant_Distance_max = total_df.groupby(['Merchant_id'])['Distance'].max().reset_index()
Merchant_Distance_max.columns = ['Merchant_id','Merchant_Distance_max']

Merchant_Distance_min = total_df.groupby(['Merchant_id'])['Distance'].min().reset_index()
Merchant_Distance_min.columns = ['Merchant_id','Merchant_Distance_min']


Coupon_Distance_mean = total_df.groupby(['Coupon_id'])['Distance'].mean().reset_index()
Coupon_Distance_mean.columns = ['Coupon_id', 'Coupon_Distance_mean']

Coupon_Distance_max = total_df.groupby(['Coupon_id'])['Distance'].max().reset_index()
Coupon_Distance_max.columns = ['Coupon_id', 'Coupon_Distance_max']

Coupon_Distance_min = total_df.groupby(['Coupon_id'])['Distance'].min().reset_index()
Coupon_Distance_min.columns = ['Coupon_id', 'Coupon_Distance_min']
total_df = pd.merge(total_df,Merchant_Distance_mean,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_Distance_max,on = ['Merchant_id'], how = 'left')
total_df = pd.merge(total_df,Merchant_Distance_min,on = ['Merchant_id'], how = 'left')

total_df = pd.merge(total_df,Coupon_Distance_mean, on = ['Coupon_id'], how = 'left')
total_df = pd.merge(total_df,Coupon_Distance_max, on = ['Coupon_id'], how = 'left')
total_df = pd.merge(total_df,Coupon_Distance_min, on = ['Coupon_id'], how = 'left')
#Scale the features
total_df = total_df.drop(['Merchant_id','Coupon_id','Day_received','Month_received','User_id','Date_received'],axis = 1)
for col in total_df.columns:
    if (col != 'Distance_Accepted_rate') & (col != 'DiscountBound_Accepted_rate'):
        total_df[col] = MinMaxScaler().fit_transform(total_df[col].values.reshape(-1,1))
#One hot encoding the discount type
total_df = pd.get_dummies(total_df, columns = ['DiscountType'], prefix = "DiscountType")
print(f' final shape of total data : {total_df.shape} ')
#Model training & validation
train_x = total_df[:len(train_df)]
test_x = total_df[len(train_df):]

#Tune the Parameter(參數調整)
def tuneParamsRandom(classifier, params, train_x, train_y, cv = 5):
    rs = RandomizedSearchCV(classifier, params, n_iter = 20, scoring = 'roc_auc', n_jobs = -1, verbose = 0, cv=cv)
    rs.fit(train_x, train_y)
    return rs.best_params_, abs(rs.best_score_)
etParams = {'n_estimators':np.arange(100,1100,50), 'max_depth':np.arange(3,11,2),
            'min_samples_leaf':np.arange(1,6,2) }

rfParams = {'n_estimators':np.arange(100,1100,50), 'max_depth':np.arange(3,11,2),
            'min_samples_split':np.arange(2,20,2), 'min_samples_leaf':np.arange(1,6,2) }

abParams = {'n_estimators':np.arange(100,1100,50),'learning_rate':np.arange(0.01,0.2,0.05)}


lgbmcParams = {'n_estimators' : np.arange(400,1200,100), 'learning_rate' : np.arange(0.01,0.1,0.02),
               'num_leaves' : np.arange(2,48,4), 'max_depth' : np.arange(3,10,2),
               'subsample' : np.arange(0.3,0.8,0.1) }

gbcParams = {'n_estimators' : np.arange(400,1200,100) , 'learning_rate' : np.arange(0.01,0.1,0.02) 
             , 'min_samples_split' : np.arange(2,30,5), 'min_samples_leaf' : np.arange(2,32,4),
              'max_depth' : np.arange(3,8,2), 'subsample' : np.arange(0.3,0.8,0.1)}

xgbParams = {'max_depth':np.arange(3,8,1),'learning_rate':np.arange(0.01,0.1,0.02)
             ,'n_estimator': np.arange(1000,3000,100),'gamma':np.arange(0.01,0.1,0.02)}

lrParams = {'C':np.arange(0.01,1,0.05), 'max_iter' : np.arange(100,500,100)}

train_label.index = train_x.index
tuneSet = pd.concat([train_x,train_label], axis = 1)
trainSet = tuneSet.sample(frac=0.5)
x_train,x_test,y_train,y_test= train_test_split(trainSet[trainSet.columns[trainSet.columns != 'label']],trainSet['label'], test_size = 0.3, random_state = 1234)
#經過測試LGB分數最高 因此其他的省略
lgbmc_best_Params, lgbmc_best_score = tuneParamsRandom(LGBMClassifier(),lgbmcParams,x_train,y_train)
print("LGBMClassifier:",lgbmc_best_Params,lgbmc_best_score)
lgbmc=LGBMClassifier(subsample=0.5,num_leaves=46,n_estimators=500,max_depth=7,learning_rate=0.03)
lgbmc.fit(train_x,train_label)
lgbmc_pred = lgbmc.predict_proba(test_x)[:,1]

#預測提交資料
sub=pd.DataFrame({'label':lgbmc_pred})
sub=pd.concat((test_id,sub),axis=1)
sub["User_id"] =sub["User_id"].apply(lambda x:str(int(x)))
sub["Coupon_id"] = sub["Coupon_id"].apply(lambda x:str(int(x)))
sub["Date_received"] = sub["Date_received"].apply(lambda x:str(int(x)))
sub['uid']=sub[["User_id", "Coupon_id", "Date_received"]].apply(lambda x:"_".join(x),axis=1)
out=sub.groupby('uid',as_index=False).mean()
out.to_csv('0619Final.csv',index=False)

