import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets
np.random.seed(5)
toy=datasets.make_blobs(centers=3,n_features=4)
X=toy[0]
y=toy[1]
estimators=[('hc_ward', AgglomerativeClustering(n_clusters=3, linkage="ward")),
            ('hc_complete', AgglomerativeClustering(n_clusters=3, linkage="complete")),
            ('hc_average', AgglomerativeClustering(n_clusters=3, linkage="average"))]

#畫出3種linKage的圖片
fignum=0
titles=['ward','complate','average']
for name,est in estimators:
    fig=plt.figure(fignum,figsize=(4,3))
    ax=Axes3D(fig,rect=[0,0,.95,1], elev=48,azim=134) #elevation仰角 Azimuthal方位視角
    #fit data
    est.fit(X)
    labels=est.labels_
    ax.scatter(X[:,3],X[:,0],X[:,2],c=labels.astype(np.float),edgecolor='k')
    ax.w_xaxis.set_ticklabels([]) #不顯示座標軸
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(titles[fignum])
    ax.dist=12
    fignum+=1
#plot the ground truth
fig=plt.figure(fignum,figsize=(4,3))
ax=Axes3D(fig,rect=[0,0,0.95,1],elev=48,azim=134)
for name, label in [('cls0', 0),
                    ('cls1', 1),
                    ('cls2', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Recorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_title('Ground Truth')
ax.dist = 12


#HW
iris=datasets.load_iris()
X=iris.data
y=iris.target
est=AgglomerativeClustering(n_clusters=3,linkage='ward')
est.fit(X)
labels=est.labels_