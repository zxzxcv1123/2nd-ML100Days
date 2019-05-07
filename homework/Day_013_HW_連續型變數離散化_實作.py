import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
data_dir='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(data_dir,'application_train.csv')
df=pd.read_csv(f_app)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df:
    if df[col].dtype=='object':
        if len(df[col].unique())<3:
            df[col]=le.fit_transform(df[col])
print(df.shape)
df['CODE_GENDER'].value_counts()
#作業 找一個變數來分割
df['DAYS_EMPLOYED'].describe()
df['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)
(abs(df['DAYS_EMPLOYED'])/365).describe() #可看出工作年齡0~50歲
test=pd.DataFrame(abs(df['DAYS_EMPLOYED'])/365)
cut_rule=np.arange(0,51,step=5)
test['cut']=pd.cut(test['DAYS_EMPLOYED'],bins=cut_rule)
#作圖觀察
test['DAYS_EMPLOYED'].hist()
test['cut'].value_counts().plot.bar() 

