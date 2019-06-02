import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
#LinearRegression(回歸問題)
#讀取糖尿病資料集
diabetes=datasets.load_diabetes()
# 為方便視覺化，我們只使用資料集中的 1 個 feature (column)
x=diabetes.data[:,np.newaxis,2] #用np.newaxis會變成(442.1)非(442,)
train_X,test_X,train_Y,test_Y=train_test_split(x,diabetes.target,test_size=0.1,random_state=4)
#建立回歸模型
reg=linear_model.LinearRegression()
reg.fit(train_X,train_Y)
pred=reg.predict(test_X)
#查看模型的參數值
print('Coefficients:',reg.coef_) #[934.0543]
#MSE
print(mean_squared_error(test_Y,pred)) #2569.6928
#畫出回歸模型與實際資料的分布
plt.scatter(test_X,test_Y,color='black')
plt.plot(test_X,pred,color='blue',linewidth=3)
plt.show()

#LosgisticRegression(分類問題)
iris=datasets.load_iris()
train_X,test_X,train_Y,test_Y=train_test_split(iris.data,iris.target,test_size=0.1, random_state=4)
#建立模型
logistic=linear_model.LogisticRegression()
logistic.fit(train_X,train_Y)
pred=logistic.predict(test_X)
#顯示分數
print('Accuracy:',accuracy_score(test_Y,pred)) #0.86666

