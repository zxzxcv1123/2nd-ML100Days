import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
train=pd.read_csv(path+'train.csv')
test=pd.read_csv(path+'test.csv')
train_Y=np.log1p(train['SalePrice'])
ids=test['Id']
train=train.drop(['SalePrice','Id'],axis=1)
test=test.drop(['Id'],axis=1)
df=pd.concat([train,test])
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
    lbl = LabelEncoder() 
    lbl.fit(list(df[c].values)) 
    df[c] = lbl.transform(list(df[c].values))
    
# 由地下室面積 + 1樓面積 + 2樓面積, 計算總坪數特徵   
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
# 把剩下少數重要的類別型欄位, 做獨熱編碼 (已變成數字的欄位, 會自動跳過)
df = pd.get_dummies(df)
print(df.shape)
# 將前述轉換完畢資料 df , 重新切成 train_X, test_X
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

# 使用三種模型 : 線性迴歸 / 梯度提升機 / 隨機森林, 參數使用 Random Search 尋找
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
linear = LinearRegression(normalize=False, fit_intercept=True, copy_X=True)
gdbt = GradientBoostingRegressor(tol=0.1, subsample=0.37, n_estimators=200, max_features=20, 
                                 max_depth=6, learning_rate=0.03)
rf = RandomForestRegressor(n_estimators=300, min_samples_split=9, min_samples_leaf=10, 
                           max_features='sqrt', max_depth=8, bootstrap=False)
# 線性迴歸預測檔 (結果有部分隨機, 請以 Kaggle 計算的得分為準, 以下模型同理)
linear.fit(train_X, train_Y)
linear_pred = linear.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(linear_pred)})
sub.to_csv('house_linear.csv', index=False) 
# 梯度提升機預測檔 
gdbt.fit(train_X, train_Y)
gdbt_pred = gdbt.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(gdbt_pred)})
sub.to_csv('house_gdbt.csv', index=False)
# 隨機森林預測檔 
rf.fit(train_X, train_Y)
rf_pred = rf.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(rf_pred)})
sub.to_csv('house_rf.csv', index=False)
# 堆疊泛化套件 mlxtend
from mlxtend.regressor import StackingRegressor
# 因為 Stacking 需要以模型作為第一層的特徵來源, 因此在 StackingRegressor 中,
# 除了要設本身(第二層)的判定模型 - meta_regressor, 也必須填入第一層的單模作為編碼器 - regressors
# 這裡第二層模型(meta_regressor)的參數, 一樣也需要用 Grid/Random Search
meta_estimator=GradientBoostingRegressor(tol=10,subsample=0.44,n_estimators=100,max_features='log2', max_depth=4, learning_rate=0.1)
stacking=StackingRegressor(regressors=[linear,gdbt,rf],meta_regressor=meta_estimator)
# 堆疊泛化預測檔 : 分數會依每次執行略有出入, 但通常 Public Score(競賽中的提交分數) 會再比單模好一些
# 雖然 Public Score 有可能比 Blending 分數略差, 但是因為不用依賴仔細調整的權重參數, 競賽結束時的 Private Score, 通常會比 Blending 好
# (因為權重依賴於 Public 的分數表現), 這種在未知 / 未曝光資料的預測力提升, 就是我們所謂 "泛化能力比較好" 在競賽/專案中的含意
stacking.fit(train_X, train_Y)
stacking_pred = stacking.predict(test_X)
sub = pd.DataFrame({'Id': ids, 'SalePrice': np.expm1(stacking_pred)})
sub.to_csv('house_stacking.csv', index=False)