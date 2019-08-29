import os
import keras
#load dataset
train,test=keras.datasets.cifar10.load_data()

#X,Y獨立放進變數
x_train,y_train=train
x_test,y_test=test
#data preprocessing
x_train=x_train/255
x_test=x_test/255
#RGB轉為向量
x_train=x_train.reshape((len(x_train),-1))
x_test=x_test.reshape((len(x_test),-1))
#將目標轉為one-hot encoding
y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

def build_mlp():
    input_layer=keras.layers.Input([x_train.shape[-1]])
    x=keras.layers.Dense(512,activation='relu')(input_layer)
    x=keras.layers.Dense(256,activation='relu')(x)
    x=keras.layers.Dense(128,activation='relu')(x)
    out=keras.layers.Dense(10,activation='softmax')(x)
    
    model=keras.models.Model(inputs=[input_layer],outputs=[out])
    return model

model=build_mlp()
#用keras 内建方法檢視模型各層參數量
model.summary()

optimizer=keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=optimizer)

history=model.fit(x_train,y_train,
                  epochs=100,batch_size=256,
                  validation_data=(x_test,y_test),
                  shuffle=True)

#plot
import matplotlib.pyplot as plt
#accuracy
plt.plot(history.history['acc'],label='acc')
plt.plot(history.history['val_acc'],label='val_acc')
plt.legend()
plt.title('acc')
plt.show()

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.title('loss')
plt.show()










#HW
import keras
train,test=keras.datasets.cifar10.load_data()
#X與Y獨立放進變數
x_train,y_train=train
x_test,y_test=test
#standard
x_train=x_train/255
x_test=x_test/255
#RGB轉為向量
x_train=x_train.reshape(50000,-1)
x_test=x_test.reshape(10000,-1)

#one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
#建立模型
def build_mlp():
    inputs=keras.layers.Input([x_train.shape[-1]])
    x=keras.layers.Dense(512,activation='relu')(inputs)
    x=keras.layers.Dense(256,activation='relu')(x)
    x=keras.layers.Dense(128,activation='relu')(x)
    out=keras.layers.Dense(10,activation='softmax')(x)
    
    model=keras.models.Model(inputs=[inputs],outputs=[out])
    return model

model=build_mlp()
#compile
model.compile(optimizer='Adam',
              loss='categorical_crossentropy'
              ,metrics=['accuracy'])
#epcoh
history=model.fit(x_train,y_train,batch_size=256,epochs=50,verbose=-1,
                  validation_data=(x_test,y_test))

import matplotlib.pyplot as plt
plt.plot(history.history['acc'],label='acc')
plt.plot(history.history['val_acc'],label='val_acc')
plt.legend()
plt.show()

plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()


