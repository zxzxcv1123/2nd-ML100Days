import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'],
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])

#延縱軸合併
result=pd.concat([df1,df2,df3])
#延橫軸合併
result2=pd.concat([df1,df4],axis=1) #補na
result3=pd.merge(df1,df4,how='inner') #取共通處合併

df5=df1.melt()#將欄轉成列
#作業
path='C:/Users/user/Desktop/JupyterNotebook/'
f_app=os.path.join(path,'application_train.csv')
df=pd.read_csv(f_app)
#1.依照 CNT_CHILDREN分成0,1~2,3~5,>5
cut_rule=[-1,0,2,5,df['CNT_CHILDREN'].max()]
df['CNT_CHILDREN_GROUP']=pd.cut(df['CNT_CHILDREN'],cut_rule,labels=['0','1~2','3~5','>5'])
df['CNT_CHILDREN_GROUP'].value_counts()
#2-1依據CNT_CHILDREN、TARGET列出各組平均AMT_INCOME_TOTAL
grouped_df=df.groupby(['CNT_CHILDREN_GROUP','TARGET'])['AMT_INCOME_TOTAL']
sub=pd.DataFrame(grouped_df.mean())
sub
#2-2繪製boxplot
df.groupby('CNT_CHILDREN_GROUP').boxplot(column='AMT_INCOME_TOTAL',by='TARGET',showfliers=False)
plt.suptitle('Average_Income')
plt.show()
#3.根據CNT_CHILDREN_GROUP、TARGET對AMT_INCOME_TOTAL經Z轉換計算分數
df['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']=grouped_df.apply(lambda x:)

df[['AMT_INCOME_TOTAL','AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']].head()

