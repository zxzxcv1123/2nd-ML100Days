import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'train.csv')
df_test=pd.read_csv(path+'test.csv')
train_Y = np.log1p(df_train['SalePrice'])
ids = df_test['Id']
df_train = df_train.drop(['Id', 'SalePrice'] , axis=1)
df_test = df_test.drop(['Id'] , axis=1)
df = pd.concat([df_train, df_test])
# 檢查 DataFrame 空缺值的狀態
def na_check(df_data):
    data_na = (df_data.isnull().sum() / len(df_data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    display(missing_data.head(10))
na_check(df)
# 部分欄位缺值補 'None'
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'FireplaceQu', 'FireplaceQu', 'FireplaceQu', 
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
             'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Functional', 'MSSubClass']
for col in none_cols:
    df[col] = df[col].fillna("None")
# 部分欄位缺值填補 0
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    df[col] = df[col].fillna(0)
# 部分欄位缺值補眾數
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
for col in mode_cols:
    df[col] = df[col].fillna(df[col].mode()[0])   
# 'LotFrontage' 有空缺時, 以同一區 (Neighborhood) 的 LotFrontage 中位數填補 (可以視為填補一種群聚編碼 )
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Utilities 參考資訊很少, 所以直接捨棄
df = df.drop(['Utilities'], axis=1)

# 四個數值欄位, 因為相異值有限, 轉成文字
label_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
for col in mode_cols:
    df[col] = df[col].astype(str)
# 相異值不太具有代表性的, 做標籤編碼
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl=LabelEncoder()
    df[c]=lbl.fit_transform(df[c])
#由地下室面積+1樓面積+2樓面積,計算總坪數特徵   
df['TotalSF']=df['TotalBsmtSF']+df['1stFlrSF']+df['2ndFlrSF']    
#把剩下的做獨熱編碼
df=pd.get_dummies(df)
print(df.shape)
#將前述轉換完畢資料df,重新切成train_X,test_X
train_num=train_Y.shape[0]
train_X=df[:train_num]
test_X=df[train_num:]
#使用三種模型:線性迴歸/梯度提升機/隨機森林,參數使用Random Search尋找
linear=LinearRegression()
gdbt=GradientBoostingRegressor(tol=0.1,subsample=0.37,n_estimators=200,max_features=20,max_depth=6,learning_rate=0.03)
rf=RandomForestRegressor(n_estimators=300,min_samples_split=9,min_samples_leaf=10,max_features='sqrt', max_depth=8, bootstrap=False)
#線性回歸模型
linear.fit(train_X,train_Y)
linear_pred=linear.predict(test_X)
sub=pd.DataFrame({'Id':ids,'SalePrice':np.expm1(linear_pred)})
sub.to_csv('house_linear.csv',index=False) #score:0.13765
# 梯度提升機預測檔 
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(gdbt_pred)})
sub.to_csv('house_gdbt.csv', index=False) #0.12911
# 隨機森林預測檔
rf.fit(train_X, train_Y)
rf_pred = rf.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(rf_pred)})
sub.to_csv('house_rf.csv', index=False) #0.165

# 混合泛化預測檔 (依 Kaggle 傳回分數調整比重, 越準確者比重越高, 依資料性質有所不同)
blending_pred = linear_pred*0.30 + gdbt_pred*0.67 + rf_pred*0.03
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(blending_pred)})
sub.to_csv('house_blending.csv', index=False) #0.1258