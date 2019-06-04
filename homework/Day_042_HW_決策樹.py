from sklearn import datasets,metrics
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import mean
iris=datasets.load_iris()
train_X,test_X,train_Y,test_Y=train_test_split(iris.data,iris.target,test_size=0.25, random_state=4)
#建立分類樹
clf=DecisionTreeClassifier()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
print(metrics.accuracy_score(test_Y,pred)) #0.9736
print("Feature importance:",clf.feature_importances_)
#Visualizing
import pydotplus 
from sklearn.tree import export_graphviz
import os
dot_data =export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf")  #
#出現bug輸入下面這行
#os.environ["PATH"]+=os.pathsep+'C:/Program Files (x86)/Graphviz2.38/bin'

#回歸樹
boston=datasets.load_boston()
train_X,test_X,train_Y,test_Y=train_test_split(boston.data,boston)
reg=DecisionTreeRegressor()
reg.fit(train_X,train_Y)
reg.feature_importances_
reg.predict(test_X)
#MSE
metrics.mean_absolute_error(test_Y,pred) #0.0263
#R2
metrics.r2_score(test_Y,pred) #0.9655
