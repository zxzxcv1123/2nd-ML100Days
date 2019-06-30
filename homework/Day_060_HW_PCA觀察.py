import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
#define PCA and subsequent LogisticRegression
logistic=SGDClassifier(loss='log',penalty='l2',max_iter=10000,tol=1e-5,random_state=0)
pca=PCA()
#Pipeline 的概念將各階段的自動化處理整合起來
pipe=Pipeline(steps=[('pca',pca),('logistic',logistic)])
#載入手寫數字辨識集
digits=datasets.load_digits()
X_digits=digits.data
y_digits=digits.target
#tune parameters
param_grid = {'pca__n_components': [4, 10, 20, 30, 40, 50, 64],
              'logistic__alpha': np.logspace(-4, 4, 5),}
search=GridSearchCV(pipe,param_grid,iid=False,cv=5,return_train_score=False)
search.fit(X_digits,y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
#繪製不同的PCA解釋度
pca.fit(X_digits)
fig,(ax0,ax1)=plt.subplots(nrows=2,sharex=True,figsize=(6,6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')
ax0.axvline(30,linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))
#繪製不同採樣點分類正確率
results=pd.DataFrame(search.cv_results_)
best_clfs=results.groupby('param_pca__n_components').apply(lambda g:g.nlargest(1,'mean_test_score'))
best_clfs.plot(x='param_pca__n_components', y='mean_test_score', yerr='std_test_score', legend=False, ax=ax1)
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')
plt.tight_layout()
plt.show()

#HW
#將參數 penalty改為L1,max_iter改為100,觀察PCA與羅吉斯迴歸做手寫辨識的效果
logistic=SGDClassifier(loss='log',
                       penalty='l1',
                       max_iter=10000,
                       tol=1e-5,
                       random_state=0)
pca=PCA()
pipe=Pipeline(steps=[('pca',pca),('logistic',logistic)])
# 先執行 GridSearchCV 跑出最佳參數
param_grid={'pca__n_components':[4,10,20,30,40,50,64],
            'logistic__alpha':np.logspace(-4,4,5)}
search=GridSearchCV(pipe,param_grid,
                    n_jobs=-1,
                    iid=False,
                    cv=5,
                    return_train_score=False)
search.fit(X_digits,y_digits)
print(search.best_params_)
pca.fit(X_digits)
# 繪製不同 components 的 PCA 解釋度
fig,(ax0,ax1)=plt.subplots(nrows=2,sharex=True,figsize=(6,6))
ax0.plot(pca.explained_variance_ratio_,linewidth=2)
ax0.set_ylabel('PCA explained variance')
ax0.axvline(30,linestyle=':', label='n_components chosen')
ax0.legend()
#繪製不同採樣點的分類正確率
results=pd.DataFrame(search.cv_results_)
best_clfs=results.groupby('param_pca__n_components').apply(lambda x:x.nlargest(1,'mean_test_score'))
best_clfs.plot(x='param_pca__n_components',y='mean_test_score',
               yerr='std_test_score', legend=False, ax=ax1)#ax為subplot的對象
ax1.set_ylabel('Classification accuracy (val)')
ax1.set_xlabel('n_components')
plt.tight_layout()
plt.show()
