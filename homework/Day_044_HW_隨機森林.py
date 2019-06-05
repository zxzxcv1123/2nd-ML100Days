from sklearn import datasets,metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
train_X,test_X,train_Y,test_Y=train_test_split(iris.data,iris.target,test_size=0.25, random_state=4)
clf=RandomForestClassifier(n_estimators=20,max_depth=4)
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
#Accuaracy
metrics.accuracy_score(test_Y,pred) #0.9736
#feature importance
print(clf.feature_importances_)

#回歸
#決策樹分數
from sklearn.ensemble import RandomForestRegressor
boston=datasets.load_boston()
train_X,test_X,train_Y,test_Y=train_test_split(boston.data,boston.target,test_size=0.25, random_state=4)
clf=RandomForestRegressor()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
#MSE
print(metrics.mean_absolute_error(test_Y,pred)) #2.6256
#R2
print(metrics.r2_score(test_Y,pred)) #0.8109
#回歸樹
from sklearn.tree import DecisionTreeRegressor
clf=DecisionTreeRegressor()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
#MSE
print(metrics.mean_absolute_error(test_Y,pred)) #3.3291
#R2
print(metrics.r2_score(test_Y,pred)) #0.7422

#隨機森林分數大於決策樹



