from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pylab import rcParams

#回歸問題
x,y=datasets.make_regression(n_features=1,random_state=42,noise=4)
model=LinearRegression()
model.fit(x,y)
pred=model.predict(x)
#MAE
mae=metrics.mean_absolute_error(pred,y) #2.8417
#MSE
mse=metrics.mean_squared_error(pred,y) #12.488
#R-square
r2=metrics.r2_score(pred,y) #0.9916

#分類問題
cancer=datasets.load_breast_cancer()
train_X,test_X,train_Y,test_Y=train_test_split(cancer.data,cancer.target,test_size=50,random_state=0)
pred=np.random.random(50) #隨機生成50比預測值
#AUC
auc=metrics.roc_auc_score(test_Y,pred)# 注意pred必須要放機率值 
print('AUC:',auc) #得到結果約 0.5，與亂猜的結果相近
#F1-score ( 2 / (1/Precision+1/Recall) ) #bata可調整，小代表看中precision，大代表看中Recall
threshold=0.5
pred_binarized=np.where(pred>threshold,1,0)#pred>0.5的值變為1，小於0.5的為0
f1=metrics.f1_score(test_Y,pred_binarized)#0.5818
precision=metrics.precision_score(test_Y,pred_binarized) #0.6666
recall=metrics.recall_score(test_Y,pred_binarized) #0.5161
#F2-score=((1+B**2)*(precision * recall)) / (B**2 * precision + recall)
f2=((1+2**2)*(precision * recall)) / (2**2*precision + recall)
print('f2-score:',f2)
