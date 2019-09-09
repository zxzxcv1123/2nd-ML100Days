import os
import keras


# 從 Keras 的內建功能中，取得 train 與 test 資料集
train, test = keras.datasets.cifar10.load_data()
## 資料前處理
def preproc_x(x, flatten=True):
    x = x / 255.
    if flatten:
        x = x.reshape((len(x), -1))
    return x

def preproc_y(y, num_classes=10):
    if y.shape[-1] == 1:
        y = keras.utils.to_categorical(y, num_classes)
    return y    
x_train, y_train = train
x_test, y_test = test

# 資料前處理 - X 標準化
x_train = preproc_x(x_train)
x_test = preproc_x(x_test)

# 資料前處理 -Y 轉成 onehot
y_train = preproc_y(y_train)
y_test = preproc_y(y_test)

def build_mlp(input_shape, output_units=10, num_neurons=[512, 256, 128]):
    input_layer = keras.layers.Input(input_shape)
    
    for i, n_units in enumerate(num_neurons):
        if i == 0:
            x = keras.layers.Dense(units=n_units, activation="relu", name="hidden_layer"+str(i+1))(input_layer)
        else:
            x = keras.layers.Dense(units=n_units, activation="relu", name="hidden_layer"+str(i+1))(x)
    
    out = keras.layers.Dense(units=output_units, activation="softmax", name="output")(x)
    
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    return model

model = build_mlp(input_shape=x_train.shape[1:])
model.summary()

# 超參數設定
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 256

optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

model.fit(x_train, y_train, 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          validation_data=(x_test, y_test), 
          shuffle=True)
#save the model
model_path='C:/Users/zxzxc/Desktop/python data/D78.h5'
model.save(model_path)
#load the mdoel
model=keras.models.load_model('D78.h5')

#plot

import matplotlib.pyplot as plt
plt.plot(model.history.history['acc'],label='acc')
plt.plot(model.history.history['val_acc'],label='val_acc')
plt.legend()
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.show()








#HW
import keras
from sklearn.preprocessing import MinMaxScaler
(train_x,train_y),(test_x,test_y)=keras.datasets.cifar10.load_data()

def preproc_x(x,flatten=True):
    x=x.reshape((len(x),-1))
    x=MinMaxScaler(feature_range=(-1,1)).fit_transform(x)
    return x
def preproc_y(y,num_classes=10):
    if y.shape[-1]==1:
        y=keras.utils.to_categorical(y,num_classes)
    return y

train_x=preproc_x(train_x)
test_x=preproc_x(test_x)
train_y=preproc_y(train_y)
test_y=preproc_y(test_y)

#建立模型
def build_mlp(input_shape,output_units=10,num_neurons=[512,256,128,64,64,32,32]):
    input_layer=keras.layers.Input(shape=input_shape)
    for i,n_units in enumerate(num_neurons):
        if i==0:
            x=keras.layers.Dense(units=n_units,activation='relu',name='hidden_layer1')(input_layer)
        else:
            x=keras.layers.Dense(units=n_units,activation='relu',name='hidden_layer'+str(i+1))(x)
    
    out=keras.layers.Dense(units=output_units,activation='softmax')(x)
   
    model=keras.models.Model(inputs=[input_layer],outputs=[out])
    return model

model=build_mlp(input_shape=train_x.shape[1:])

#compile
optimizer=keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],
          optimizer=optimizer)

#訓練模型
model.fit(train_x,train_y,
          epochs=100,
          batch_size=256,
          validation_data=(test_x,test_y),
          shuffle=True)

#save model
model.save('D78.h5')

import matplotlib.pyplot as plt

plt.plot(model.history.history['acc'],label='acc')
plt.plot(model.history.history['val_acc'],label='val_acc')
plt.legend()
plt.show()


    
    