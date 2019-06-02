import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
#糖尿病資料
diabetes=datasets.load_diabetes()
train_X,test_X,train_Y,test_Y=train_test_split(diabetes.data,diabetes.target,test_size=0.2, random_state=4)

#線性回歸模型
reg=LinearRegression()
reg.fit(train_X,train_Y)
pred=reg.predict(test_X)
print(reg.coef_) #顯示係數
#MSE
print("MSE:",mean_squared_error(test_Y,pred)) #2939.42
#R-square
print("R2:",r2_score(test_Y,pred)) #0.4610

#LASSO
lasso=Lasso(alpha=1)
lasso.fit(train_X,train_Y)
pred=lasso.predict(test_X)
#可以看到許多係數都變成0，Lasso Regression的確可以做特徵選取
print(lasso.coef_)
#MSE
print("MSE:",mean_squared_error(test_Y,pred))#3505.84
#R-square
print("R2:",r2_score(test_Y,pred)) #0.3572

#Ridge
ridge=Ridge(alpha=1)
ridge.fit(train_X,train_Y)
pred=ridge.predict(test_X)
#很明顯看到比起 Linear Regression，參數的數值都明顯小了許多
print(ridge.coef_)
#MSE
print("MSE:",mean_squared_error(test_Y,pred)) #3221.4209
#R2
print("R2:",r2_score(test_Y,pred)) #0.4093


#可以看見 LASSO 與 Ridge 的結果並沒有比原本的線性回歸來得好， 
#這是因為目標函數被加上了正規化函數，讓模型不能過於複雜，
#相當於限制模型擬和資料的能力。因此若沒有發現 Over-fitting 的情況，
#是可以不需要一開始就加上太強的正規化的。