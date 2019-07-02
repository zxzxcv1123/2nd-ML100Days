import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold,datasets
from time import time
#設定模型與繪圖參數
n_samples=300
n_components=2
(fig,subplots)=plt.subplots(2,5,figsize=(15,6))
perplexities=[4,6,9,14,21,30,45,66,100]
#設定同心圓資料點
X,y=datasets.make_circles(n_samples=300,factor=0.5,noise=0.05)
red=y==0
green=y==1
#繪製資料原圖
ax=subplots[0][0]
ax.set_title('Original')
ax.scatter(X[red, 0], X[red, 1], c="r")
ax.scatter(X[green, 0], X[green, 1], c="g")
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
# 繪製不同 perplexity 下的 t-SNE 分群圖
for i,perplexity in enumerate(perplexities): #i=[0,8]
    if i<4:
        ax = subplots[0][i+1]
    else:
        ax = subplots[1][i-4]

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[green, 0], Y[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
    
plt.show()
        
    
