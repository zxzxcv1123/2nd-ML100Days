import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import os 
import seaborn as sns
import warnings #忽略警告訊息
warnings.filterwarnings('ignore')
data_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(data_dir,'application_train.csv')
df=pd.read_csv(f_app)
df['DAYS_BIRTH']=abs(df['DAYS_BIRTH']) #取絕對值
#畫圖
plt.hist(df['DAYS_BIRTH']/365,edgecolor='k',bins=25) #edgecolor為線框顏色
plt.title('Age of Client')
plt.xlabel('Age (years)')
plt.ylabel('Count')
#改變style在執行一次上面(預設為'defult')
plt.style.use('ggplot')
plt.style.use('seaborn')
#設定繪圖區域的長與寬
plt.figure(figsize=(10,8))
#Kernel Density Estimation(KDE)plot:會準時還貸者-下紅線
sns.kdeplot(df.loc[df['TARGET']==0,"DAYS_BIRTH"]/365,label='target=0')
sns.kdeplot(df.loc[df['TARGET']==1,'DAYS_BIRTH']/365,label='target=1')
# KDE, 比較不同的 kernel function
sns.kdeplot(df.loc[df['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Gaussian esti.', kernel='gau')
sns.kdeplot(df.loc[df['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Cosine esti.', kernel='cos')
sns.kdeplot(df.loc[df['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Triangular esti.', kernel='tri')
plt.show()
#將資料依照歲數將20~70歲切成11個區間
age_data=df[['DAYS_BIRTH','TARGET']]
age_data['YEARS_BIRTH']=age_data['DAYS_BIRTH']/365
bin_cut=np.linspace(20,70,11)
age_data['YEARS_BINNED']=pd.cut(age_data['YEARS_BIRTH'],bins=bin_cut)
print(age_data['YEARS_BINNED'].value_counts())
#繪圖 &交集
year_group_sorted = sorted(age_data.YEARS_BINNED.unique())
plt.figure(figsize=(8,6))
for i in range(len(year_group_sorted)):
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 0), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
    
    sns.distplot(age_data.loc[(age_data['YEARS_BINNED'] == year_group_sorted[i]) & \
                              (age_data['TARGET'] == 1), 'YEARS_BIRTH'], label = str(year_group_sorted[i]))
plt.title('KDE with Age groups')
plt.show()
# 以年齡區間為 x, target 為 y 繪製 barplot
px=age_data['YEARS_BINNED']
py=age_data['TARGET']
sns.barplot(px,py)
# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')
