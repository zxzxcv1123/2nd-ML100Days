import keras
(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()

def preproc_x(x,flatten=True):
    x=x/255
    if flatten:
        x=x.reshape(len(x),-1)
    return x

def preproc_y(y,num_classes=10):
    if y.shape[-1]==1:
        y=keras.utils.to_categorical(y,num_classes)
    return y

x_train=preproc_x(x_train)
x_test=preproc_x(x_test)
y_train=preproc_y(y_train)
y_test=preproc_y(y_test)

def build_mlp(input_shape,output_units=10,num_neurons=[512,256,128]):
    input_layer=keras.layers.Input(input_shape)
    
    for i,n_units in enumerate(num_neurons):
        if i == 0:
            x= keras.layers.Dense(units=n_units,activation='relu',name='hidden_layer1')(input_layer)
        else:
            x=keras.layers.Dense(units=n_units,activation='relu',name='hidden_layer'+str(i+1))(x)
    
    out=keras.layers.Dense(units=output_units,activation='softmax',name='output')(x)
    model=keras.models.Model(inputs=input_layer,outputs=out)
    return model

results={}
for lr in [0.1,0.01,0.001,0.0001,0.00001]:
    keras.backend.clear_session() # 把舊的 Graph 清掉
    print('Experiment with LR={}'.format(lr))
    model=build_mlp(input_shape=x_train.shape[1:])
    model.summary()
    optimizer=keras.optimizers.SGD(lr=lr,nesterov=True,momentum=0.95)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=optimizer)
    model.fit(x_train,y_train,
              epochs=50,
              batch_size=256,
              validation_data=(x_test,y_test),
              shuffle=True)
    # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["acc"]
    valid_acc = model.history.history["val_acc"]
    
    exp_name_tag = "exp-lr-%s" % str(lr)
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}

import matplotlib.pyplot as plt
color_bar = ["r", "g", "b", "y", "m", "k"]

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond, color=color_bar[i])
plt.title("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond, color=color_bar[i])
plt.title("Accuracy")
plt.legend()
plt.show()



#HW
import keras

(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()

def preproc_x(x,flatten=True):
    x=x/255
    if flatten:
        x=x.reshape(len(x),-1)
    return x
def preproc_y(y,num_classes=10):
    if y.shape[-1]==1:
        y=keras.utils.to_categorical(y,num_classes)
    return y

x_train=preproc_x(x_train)
x_test=preproc_x(x_test)
y_train=preproc_y(y_train)
y_test=preproc_y(y_test)

def build_mlp(input_shape,output_units=10,neurons=[512,256,128]):
    input_layer=keras.layers.Input(shape=input_shape)
    for i,n_units in enumerate(neurons):
        if i==0:
            x=keras.layers.Dense(units=n_units,activation='relu',name='hidden layer1')(input_layer)
        else:
            x=keras.layers.Dense(units=n_units,activation='relu',name='hidden layer'+str(i+1))(x)
    out=keras.layers.Dense(units=output_units,activation='softmax',name='output layer')(x)
    
    model=keras.models.Model(inputs=input_layer,outputs=out)
    
    return model

"""Code Here
設定超參數
"""
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 256
MOMENTUM = [0.99, 0.95, 0.90]
"""Code Here
撰寫你的訓練流程並將結果用 dictionary 紀錄
"""
for mom in MOMENTUM:
    keras.backend.clear_session() # 把舊的 Graph 清掉
    print("Experiment with MOM = %.6f" % (mom))
    model = build_mlp(input_shape=x_train.shape[1:])
    model.summary()
    optimizer = keras.optimizers.SGD(lr=LEARNING_RATE, nesterov=True, momentum=mom)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    model.fit(x_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              validation_data=(x_test, y_test), 
              shuffle=True)
    
    # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["acc"]
    valid_acc = model.history.history["val_acc"]
    
    exp_name_tag = "exp-mom-%s" % str(mom)
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}
import matplotlib.pyplot as plt
%matplotlib inline
"""Code Here
將結果繪出
"""
color_bar = ["r", "g", "b", "y", "m", "k"]

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond, color=color_bar[i])
plt.title("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond, color=color_bar[i])
plt.title("Accuracy")
plt.legend()
plt.show()