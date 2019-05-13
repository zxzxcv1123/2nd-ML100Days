import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure,show
path='C:/Users/user/Desktop/JupyterNotebook'
app_train=os.path.join(path,'application_train.csv')
df=pd.read_csv(app_train) 
df.columns
df_describe=df.describe()
df['TARGET'].skew() #偏差 
df['TARGET'].kurt() #偏態 越大偏態越嚴重
df_corr=df.corr() #共變異矩陣
#直方圖
sns.distplot(df['DAYS_BIRTH'])
plt.hist(df['DAYS_BIRTH'])
#盒鬚圖
plt.boxplot(df['DAYS_BIRTH'])
sns.boxplot(df['DAYS_BIRTH'])
