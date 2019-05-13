import pandas as pd 
import numpy as np
path='C:/Users/user/Desktop/JupyterNotebook/DATA2/'
df_train=pd.read_csv(path+'train.csv')
df_test=pd.read_csv(path+'test.csv')
df_train.shape
#抽離SalePrice、Ids之後合併資料
train_Y=np.log1p(df_train['SalePrice'])
ids=df_test['Id']
df_train=df_train.drop(['SalePrice','Id'],axis=1)
df_test=df_test.drop(['Id'],axis=1)
df=pd.concat([df_test,df_train])
# 秀出資料欄位的類型, 與對應的數量
dtype_df=df.dtypes.reset_index()
dtype_df.columns=['Count','Column Type']
dtype_df=dtype_df.groupby('Column Type').aggregate('count').reset_index()
#將欄位名稱存於三個list中
int_feature=[]
float_feature=[]
object_feature=[]
for dtype,feature in zip(df.dtypes,df.columns):
    if dtype=='int64':
        int_feature.append(feature)
    elif dtype=='float64':
        float_feature.append(feature)
    else:
        object_feature.append(feature)
#f-string為最新用法 取代format
print(f'{len(int_feature)} Integer Features : {int_feature}\n')
print(f'{len(float_feature)} Float Features : {float_feature}\n')
print(f'{len(object_feature)} Object Features : {object_feature}')


#作業1三種類型int、float、object)的欄位
#分別進行mean、Max、nunique那些會發生問題?
#數值型
df[int_feature].mean()
df[int_feature].max()
df[int_feature].nunique() #沒錯誤，但不適用
#數值型
df[float_feature].mean()
df[float_feature].max()
df[float_feature].nunique() #沒錯誤，但不適用
#類別型
df[object_feature].mean() #無法
df[object_feature].max() #無法
df[object_feature].nunique()

