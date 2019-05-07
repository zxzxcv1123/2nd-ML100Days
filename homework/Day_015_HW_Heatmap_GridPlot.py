import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
df_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(df_dir,'application_train.csv')
df=pd.read_csv(f_app)
# 取出 EXT_SOURCE 的幾項變數並檢驗其相關性
df['DAYS_BIRTH']=abs(df['DAYS_BIRTH'])
ext_data=df[['TARGET','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH']]
ext_data_corrs=ext_data.corr()
# 繪製相關係數 (correlations) 的 Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(ext_data_corrs,cmap = plt.cm.RdYlBu_r,annot=True) #cmap為顏色，annot顯示係數
plt.title('Correlation Heatmap')
#檢查三個變數在TARGET上的分布
plt.figure(figsize=(8,24))
for i,source in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):
   #做subplot
    plt.subplot(3,1,i+1)
    #KDE圖形
    sns.kdeplot(df.loc[df['TARGET']==0,source],label='TARGET=0')
    sns.kdeplot(df.loc[df['TARGET']==1,source],label='TARGET=1')
    #加上標籤
    plt.title(source)
    plt.xlabel(source);plt.ylabel('Density')

#將day轉換成Year
plot_data=ext_data.copy()
plot_data['YEARS_BIRTH']=plot_data['DAYS_BIRTH']/365
plot_data.drop(['DAYS_BIRTH'],axis=1,inplace=True)
#把NaN數值刪去,並限制資料上限為 1000如果點太多，會畫很久!
N_sample=1000
plot_data=plot_data.dropna().sample(n=N_sample)
#建立pairgrid物件
grid=sns.PairGrid(data=plot_data,size=2,diag_sharey=False,
                  hue='TARGET',
                  vars=[x for x in list(plot_data.columns)if x !='TARGET'])
#上半部為scatter
grid.map_upper(plt.scatter,alpha=0.5)
#對角線為histogram
grid.map_diag(sns.kdeplot)
#下半部為desity plot
grid.map_lower(sns.kdeplot,cmap=plt.cm.OrRd_r)


#作業1 建立10*10 -1~1的矩陣並繪製Heatmap
x=np.random.uniform(-1,1,100) #-1~1隨機採100個
matrix=x.reshape(10,10)
plt.figure(figsize=(10,10))
heatmap=sns.heatmap(matrix,cmap=plt.cm.RdYlBu_r)
plt.show()

#作業2 1000*3 -1~1 做PairPlot(上半部scatter,對角線為hist,下半部density)
nrow=1000
ncol=3
matrix=np.random.uniform(-1,1,[nrow,ncol])
#隨機給予0,1,2三種標籤，並指定為index
indice=np.random.choice([0,1,2],size=nrow) #0,1,2這三個數隨機創造1000個
plot_data=pd.DataFrame(matrix,indice)
#作圖
grid=sns.PairGrid(data=plot_data,size=3,diag_sharey=False)
grid.map_upper(plt.scatter,alpha=0.2)
grid.map_diag(sns.distplot)
grid.map_lower(sns.kdeplot,cmap=plt.cm.OrRd_r)
plt.show()

#作業3 1000*3常態分佈，做PairPlot(上半部scatter,對角線為hist,下半部density)
matrix=np.random.randn(3000).reshape(1000,3)
indice=np.random.choice([0,1,2],size=1000)
plot_data=pd.DataFrame(data=matrix,index=indice)

grid=sns.PairGrid(data=plot_data,size=3,diag_sharey=False)
grid.map_upper(plt.scatter)
grid.map_diag(sns.distplot)
grid.map_lower(sns.kdeplot,cmap=plt.cm.OrRd_r)