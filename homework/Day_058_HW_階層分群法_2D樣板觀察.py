import time 
import warnings 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster ,datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle,islice

#設定2D樣板資料
n_samples=1500
random_state=100
#生成 同心圓 資料點
noisy_circles=datasets.make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
#生成 稀疏三群 資料點(使用轉換矩陣)(原本為緊密的三群)
X,y=datasets.make_blobs(n_samples=n_samples,random_state=random_state)
transformation=[[0.6, -0.6], [-0.4, 0.8]] 
X_aniso = np.dot(X, transformation)#第一欄X1*0.6-X2*0.6 第二欄X1*-0.4+X2*0.8
aniso = (X_aniso, y)
#生成 斜向三群 資料點
varied=datasets.make_blobs(n_samples=n_samples,cluster_std=[1,2.5,0.5],random_state=random_state)
#設定群集與繪圖參數
plt.figure(figsize=(10,8))
plot_num=1
default_base={'n_neighbors':10,"n_clusters":3}
datasets=[(noisy_circles,{'n_clusters':2}),(varied, {'n_neighbors': 2}), (aniso, {'n_neighbors': 2})]
#Perform drawing loops of various sample matetials
for i_dataset,(dataset,algo_params) in enumerate(datasets):
    #copy parameter and stadardization
    params=default_base.copy()
    params.update(algo_params)
    X,y=dataset
    X=StandardScaler().fit_transform(X)
    #set up Hierarchical clustering with three different parameters
    ward=cluster.AgglomerativeClustering(n_clusters=params['n_clusters'],linkage='ward')
    complete = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="complete")
    average = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="average")
    clustering_algorithms=(
            ('Average Linkage',average),
            ("Complete Linkage",complete),
            ('Ward Linkage',ward))
    #Draw three kind of graphics
    for name,algorithm in clustering_algorithms:
        t0=time.time()
        algorithm.fit(X)
        t1=time.time()
        #param(object,name) return True,False
        if hasattr(algorithm,'labels_'):
            y_pred=algorithm.labels_.astype(np.int)
        else:
            y_pred=algorithm.predict(X)
        plt.subplot(len(datasets),len(clustering_algorithms),plot_num)
        if i_dataset==0:
            plt.title(name,size=18)
        colors=np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                           '#f781bf', '#a65628', '#984ea3',
                                           '#999999', '#e41a1c', '#dede00']),
                                    int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
plt.show()
    
    
#HW
# 設定 2D 樣板資料
n_samples = 1500
random_state = 100
# 生成 雙半月 資料點
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# 生成 緊密三群 資料點
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
# 生成 2維均勻分布 資料點
no_structure = np.random.rand(n_samples, 2), None #rand:Return samples from the “standard normal” distribution.
#set cluster and plot parameters
# 設定群集與繪圖參數
plt.figure(figsize=(10, 8))
plot_num = 1
default_base = {'n_neighbors': 10, 'n_clusters': 3}
datasets = [(noisy_moons, {'n_clusters': 2}), (blobs, {}), (no_structure, {})]
for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # 複製參數與標準化
    params = default_base.copy()
    params.update(algo_params)    
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # 設定三種不同參數的 Hierarchical clustering
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="ward")
    complete = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="complete")
    average = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage="average")
    clustering_algorithms = (
        ('Average Linkage', average),
        ('Complete Linkage', complete),
        ('Ward Linkage', ward))

    # 繪製三種圖形
    for name, algorithm in clustering_algorithms:
        # t0 / t1 : 紀錄時間差
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'), transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1
plt.show()
    
    