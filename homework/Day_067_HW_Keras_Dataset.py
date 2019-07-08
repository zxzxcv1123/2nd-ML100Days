import numpy as np
from keras.datasets import cifar10
np.random.seed(10)
#CIFAR10 50000張32*32彩色訓練圖像
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print('train:',len(x_train))
print('test:',len(x_test))
#查詢維度資訊
x_train.shape
#search data dimension
y_train.shape
#針對物件圖像數據集的類別編列成字典
label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
#導入影像列印模組
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,25):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([]) #不顯示座標軸  
        idx+=1 
    plt.show()

#針對不同的影像作標記
plot_images_labels_prediction(x_train,y_train,[],0)        

#影像正規化
x_train[0][0][0] #array([59,62,63])
x_train_normalize=x_train.astype('float32')/255.0
x_test_normalize=x_test.astype('float32')/255.0
x_train_normalize[0][0][0] #array([0.23137255,0.24313726,0.24705882]

#轉換label為OneHot Encoding
y_train.shape #(50000,1)
y_train[:5]
from keras.utils import np_utils
y_train_onehot=np_utils.to_categorical(y_train)

y_train_onehot.shape #(50000,10)
y_train_onehot[:,5]


#HW
import numpy as np
from keras.datasets import cifar100
import matplotlib.pyplot as plt
np.random.seed(100)

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
plt.imshow(x_train[0])
plt.xticks([])
plt.yticks([])
#image normalize
x_train_normalize=x_train.astype(np.float32)/255.0
x_test_normalize=x_test.astype(np.float32)/255.0
#對label 做 OneHotEncoding
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
y_train_onehot=np_utils.to_categorical(y_train)
y_test_onehot=np_utils.to_categorical(y_test)
y_train[0]
y_train_onehot[0]
onehot=OneHotEncoder()
test_test=onehot.fit_transform(y_train)
