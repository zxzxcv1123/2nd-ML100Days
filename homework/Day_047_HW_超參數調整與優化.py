from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
# 讀取波士頓房價資料集
boston=datasets.load_boston()
# 切分訓練集/測試集
train_X,test_X,train_Y,test_Y=train_test_split(boston.data,boston.target,test_size=0.25, random_state=42)
# 建立模型
clf=GradientBoostingRegressor(random_state=7)
# 先看看使用預設參數得到的結果，約為 8.379 的 MSE
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
print(metrics.mean_squared_error(test_Y,pred))

#設定要訓練的超參數組合
n_estimators=[100,200,300]
max_depth=[1,3,5]
param_grid=dict(n_estimators=n_estimators,max_depth=max_depth)
# 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1會使用全部cpu平行運算)
grid_search=GridSearchCV(clf,param_grid,scoring='neg_mean_squared_error',n_jobs=-1,verbose=1)
# 開始搜尋最佳參數
grid_result=grid_search.fit(train_X,train_Y)
#印出最佳結果與最佳參數
print(f'Best Accuracy:{grid_result.best_score_}')
print(f'參數:{grid_result.best_params_}')

#使用最佳參數重新建立模型
clf_best=GradientBoostingRegressor(max_depth=3,n_estimators=200)
clf_best.fit(train_X,train_Y)
pred=clf_best.predict(test_X)
# 調整參數後約可降至 8.30 的 MSE
print(metrics.mean_squared_error(test_Y,pred)


#practice
from sklearn.ensemble import RandomForestClassifier
iris=datasets.load_iris()
train_X,test_X,train_Y,test_Y=train_test_split(iris.data,iris.target,test_size=0.2)
clf=RandomForestClassifier()
clf.fit(train_X,train_Y)
pred=clf.predict(test_X)
metrics.accuracy_score(test_Y,pred) #0.9666
#設定要訓練的超參數組合
n_estimators=[100,200,300]
max_depth=[1,3,5,7]
max_features=['auto','sqrt']
min_samples_split=[2,5,10]
grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split}
#建立搜尋物件
grid_search=GridSearchCV(clf,grid,n_jobs=-1,verbose=1)
#開始搜尋
grid_result=grid_search.fit(train_X,train_Y)
best_param=grid_result.best_params_
clf_best=RandomForestClassifier(n_estimators=100,max_depth=3,max_features='sqrt',min_samples_split=5)
clf_best.fit(train_X,train_Y)
pred=clf_best.predict(test_X)
metrics.accuracy_score(test_Y,pred) #1
