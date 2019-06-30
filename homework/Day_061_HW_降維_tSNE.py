import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold,datasets
#load digits (4個數字) 資料集
digits=datasets.load_digits(n_class=4)
X=digits.data
y=digits.target
#set param
n_samples,n_features=X.shape
n_neighbors=30
tsne=manifold.TSNE(n_components=2,random_state=0,init='pca',
                   learning_rate=200,early_exaggeration=12)
#資料建模 並 視覺化
def plot_embedding(X,title=None):
    x_min,x_max=np.min(X,0),np.max(X,0)
    X=(X-x_min)/(x_max-x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'): #object是否包含屬性
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1,1]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images)**2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")

X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the digits")

plt.show()
