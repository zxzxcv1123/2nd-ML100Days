from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist
np.random.seed(10)
#載入手寫辨識的資料集
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#指定測試集與訓練資料集
X_train=x_train.reshape(60000,784).astype('float32')
X_test=x_test.reshape(10000,784).astype('float32')
#normalize inputs from 0-255 to 0-1
x_train_normalize=X_train/255
x_test_normalize=X_test/255
#one hot
y_train_onehot=np_utils.to_categorical(y_train)
y_test_onehot=np_utils.to_categorical(y_test)

#建立模型
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=256,
                input_dim=784,
                kernel_initializer='normal',
                activation='relu'))
#建構輸出層
model.add(Dense(units=10,
                kernel_initializer='normal',
                activation='softmax'))
print(model.summary())

#訓練模型
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=x_train_normalize,
                        y=y_train_onehot,validation_split=0.2,
                        epochs=10,batch_size=32,verbose=-1)
#plot graph
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='best')
    plt.show()

train_history.history

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#assess model accuracy
scores=model.evaluate(x_test_normalize,y_test_onehot)
print('accuracy:',scores[1]) #0.9786

#HW  多加一層 查看準確率有甚麼差別
from keras.utils import np_utils
import numpy as np
np.random.seed(10)
#laod data
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#reshape and normalize
x_train=x_train.reshape(60000,784).astype('float32')/255
x_test=x_test.reshape(10000,784).astype('float32')/255
#OnehotEncoding label data
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#creat a model
from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=256,input_dim=784,
                kernel_initializer='normal',#權重的初始值
                activation='relu'))
model.add(Dense(units=10,activation='softmax'))
#training model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
model_history=model.fit(x=x_train,
                        y=y_train,validation_split=0.2,
                        epochs=10,batch_size=32,verbose=-1)
#show train process
import matplotlib.pyplot as plt
def show_image(model_history,train,validation):
    plt.plot(model_history.history[train])
    plt.plot(model_history.history[validation])
    plt.title('train processing')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['trian','validation'],loc='best')
    plt.show()
show_image(model_history,'acc','val_acc')
show_image(model_history,'loss','val_loss')
#evaluate model accuracy
scores=model.evaluate(x_test,y_test)
print('accuracy:',scores[1]) #0.9741
