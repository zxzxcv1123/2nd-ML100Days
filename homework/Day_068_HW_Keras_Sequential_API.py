import keras 
from keras.datasets import cifar10 
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
batch_size=32
num_classses=10
epochs=10 #整個資料完整run過的次數

#the data,shuffled and split between train and test sets:
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print('x_train_shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')
#convert class vetors to binary class matrics
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)
#建立線性堆疊模型
#Sequential的第一層需要加入input_shape參數
model=Sequential() 
#Conv2D layer(卷基層)
#卷積核的數目=64(即輸出維度) 
#kernel_size=(3*3)(width and lenth)(圖片切分的數量=9)
#“same”out has the same lenth as the original input(reetain boundary data)
#input_shape(32,32,3)it means 32*32 color RGB image
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=x_train.shape[1:]))

#對上層輸出應用 激活函數
model.add(Activation('relu'))
#Flatten:對上層輸出 一維化
model.add(Flatten())
#Dense實現全連階層
model.add(Dense(units=512)) #輸出維度=512
#Activation:對上層輸出應用 繳活函數
model.add(Activation('relu'))
#dropout:對上層輸出應用 dropout 以防止過擬合
model.add(Dropout(0.5))
#10-way softmax layer 
#which means it will return an array of 10 probability scores
model.add(Dense(num_classses)) 
model.add(Activation('softmax'))

print(model.summary())



#HW
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train.shape
# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=x_train.shape[1:]))
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
model.add(Dropout(0.25))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(128, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
model.add(Dropout(0.5))
# 使用 softmax activation function，將結果分類
model.add(Dense(10, activation='softmax'))

print(model.summary())
