"""
請結合前面的知識與程式碼，比較不同的 optimizer 與 learning rate 組合對訓練的結果與影響
常見的 optimizer 包含
SGD
RMSprop
AdaGrad
Adam
"""
import keras
(train_x,train_y),(test_x,test_y)=keras.datasets.cifar10.load_data()
## 資料前處理
train_x=(train_x/255).reshape(len(train_x),-1)
test_x=(test_x/255).reshape(len(test_x),-1)
train_y=keras.utils.to_categorical(train_y,10)
test_y=keras.utils.to_categorical(test_y,10)

def build_mlp(input_shape,output_units=10,num_neurons=[512,256,128]):
    input_layer=keras.layers.Input(input_shape)
    for i,n_units in enumerate(num_neurons):
        if i==0:
            x=keras.layers.Dense(units=n_units,activation='relu',
                                 name='hidden_layer'+str(i+1))(input_layer)
        else:
            x=keras.layers.Dense(units=n_units,activation='relu',
                                 name='hidden_layer'+str(i+1))(x)
    out=keras.layers.Dense(units=output_units,activation='softmax',
                           name='output_layer')(x)
    model=keras.models.Model(inputs=[input_layer],outputs=[out])
    
    return model


## 超參數設定
"""
Set your required experiment parameters
"""
learning_rate=0.01
epochs=30
batch_size=512
momentum=0.9


results = {}
"""
建立你的訓練與實驗迴圈並蒐集資料
"""
for name in ['SGD','RMSprop','AdaGrad','Adam']:
    keras.backend.clear_session()# 把舊的 Graph 清掉
    print('Experiment with optimizer'+name)
    model=build_mlp(input_shape=train_x.shape[1:])
    model.summary()
    if name=='SGD':
        optimizer=keras.optimizers.SGD()
    elif name=='RMSprop':
        optimizer=keras.optimizers.RMSprop()
    elif name=='RMSprop':
        optimizer=keras.optimizers.RMSprop()
    else:
        optimizer=keras.optimizers.Adam()
    
    model.compile(optimizer=optimizer,loss='categorical_crossentropy'
                  ,metrics=['accuracy'])
    
    model.fit(train_x,train_y,batch_size=batch_size,epochs=30,validation_data=(test_x,test_y))
    
       # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["acc"]
    valid_acc = model.history.history["val_acc"]
    
    exp_name_tag = "exp-optimizer-{}".format(name)
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}
import matplotlib.pyplot as plt

"""
將實驗結果繪出
"""
color=["r", "g", "b", "y", "m", "k"]
plt.figure(figsize=(8,6))
for i,cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond, color=color[i])
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond, color=color[i])
plt.title("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond, color=color[i])
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond, color=color[i])
plt.title("Accuracy")
plt.legend()
plt.show()