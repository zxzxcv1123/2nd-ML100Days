import keras
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#CREATE MODEL
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
#宣告使用序列模型
model=Sequential()
#卷積層1與池化層1
model.add(Conv2D(filters=32,kernel_size=(3,3), #輸出32,切割數3*3
                 input_shape=(32,32,3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
#卷積層2與池化層2
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
#卷積層3與池化層3
model.add(Conv2D(filters=128,kernel_size=(3,3),
                 activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
#卷積層4與池化層4
model.add(Conv2D(filters=256,kernel_size=(3,3),
                 activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
#建立神經網路(平坦層、隱藏層、輸出層)
model.add(Flatten())
#建立全網路連接層
model.add(Dense(512,activation='relu'))
model.add(Dense(512,activation='relu'))
#建立輸出層
model.add(Dense(10,activation='softmax'))
#檢查model 的STACK
model.summary()
#模型編譯
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',metrics=['accuracy'])
#模型訓練，'train_history'把訓練過程所得到的數值存起來
train_history=model.fit(x=x_train,y=y_train,validation_split=0.25,
                        epochs=12,batch_size=128,verbose=-1)
import matplotlib.pyplot as plt
def show_image(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()
show_image('acc','val_acc')


#HW
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#modeling
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D,ZeroPadding2D
from keras.models import Sequential
model=Sequential()
#卷積層1
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=x_train.shape[1:],
                 activation='relu',padding='same'))
model.add(Dropout(rate=0.25)) #防止過度擬合
model.add(MaxPooling2D(pool_size=(2,2)))
#卷積層2
model.add(Conv2D(filters=64,kernel_size=(3,3),
                 padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
#一維化
model.add(Flatten())
model.add(Dropout(rate=0.25))
#隱藏層
model.add(Dense(activation='relu',units=1024))
model.add(Dropout(rate=0.25))
#輸出層
model.add(Dense(10,activation='softmax'))

#training model
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',metrics=['accuracy'])
train_history=model.fit(x=x_train,y=y_train,batch_size=32,epochs=12,
                        verbose=-1,validation_split=0.2)
import matplotlib.pyplot as plt
def show_image(acc,val_acc):
    plt.plot(train_history.history[acc])
    plt.plot(train_history.history[val_acc])
    plt.ylabel('train')
    plt.xlabel('epochs')
    plt.title('train processing')
    plt.legend(['train','validation'],loc='best')
show_image('acc','val_acc')
