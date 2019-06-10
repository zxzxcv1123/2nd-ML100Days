from sklearn import datasets,metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
iris=datasets.load_iris()
train_X,test_X,train_Y,test_Y=train_test_split(iris.data,iris.target,test_size=0.25,random_state=4)
clf=GradientBoostingClassifier()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
print('Accuracy:',metrics.accuracy_score(test_Y,pred)) #0.9736
#手寫辨識資料
digits=datasets.load_digits()
train_X,test_X,train_Y,test_Y=train_test_split(digits.data,digits.target,test_size=0.25,random_state=4)
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
print('Accuracy:',metrics.accuracy_score(test_Y,pred)) #0.96666
