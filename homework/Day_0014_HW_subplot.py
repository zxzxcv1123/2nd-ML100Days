import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
df_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(df_dir,'application_train.csv')
df=pd.read_csv(f_app)
# 根據年齡分成不同組別 (年齡區間 - 還款與否)
age_data=df[['TARGET','DAYS_BIRTH']]
age_data['YEARS_BIRTH']=abs(age_data['DAYS_BIRTH'])/365
#離散畫，20~70切11個點(10組)
age_data['YEARS_BINNED']=pd.cut(abs(age_data['DAYS_BIRTH'])/365,bins=np.linspace(20,70,num=11))
print(age_data['YEARS_BINNED'].value_counts())
#資料分組後排序
year_group_sorted=np.sort(age_data['YEARS_BINNED'].unique())

# 繪製分群後的 10 條 KDE 曲線 &代表交集
plt.figure(figsize=(8,5))
for i in range(10):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED']==year_group_sorted[i])&
                              (age_data['TARGET']==0),'YEARS_BIRTH'],label=str(year_group_sorted[i]))
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED']==year_group_sorted[i])&
                              (age_data['TARGET']==1),'YEARS_BIRTH'],label=str(year_group_sorted[i]))
plt.title('KDE with Age groups')

#每張圖片大小8*8
plt.figure(figsize=(8,8))
#plt.subplot 三碼如上所述, 分別表示 row總數, column總數, 本圖示第幾幅(idx)
plt.subplot(211)
plt.plot([0,1],[0,1],label='I am subplot1')
plt.legend()

plt.subplot(212)
plt.plot([1,0],[0,1],label='I am subplot2')
plt.legend()
plt.show()
#subplot index 超過10以上的繪製方式
plt.figure(figsize=(10,30))
for i in range(10):
    plt.subplot(5,2,i+1)
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED']==year_group_sorted[i])&
                               (age_data['TARGET']==0),'YEARS_BIRTH'],
    label='TARGET=0',hist=False)
    plt.subplot(5,2,i+1)
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED']==year_group_sorted[i])&
                               (age_data['TARGET']==1),'YEARS_BIRTH'],
    label='TARGET=1',hist=False)
    plt.title=str(year_group_sorted[i])
#根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram          
df['HOUSETYPE_MODE'].unique() #觀察資料
df['AMT_CREDIT'].describe() #觀察資料
unique_house_type=df['HOUSETYPE_MODE'].unique()
nrows=len(unique_house_type)
ncols=nrows//2  #float:x/y int:x//y

plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows,ncols,i+1)
    df.loc[(df['HOUSETYPE_MODE']==unique_house_type[i]),'AMT_CREDIT'].hist()
    plt.title(str(unique_house_type[i]))






