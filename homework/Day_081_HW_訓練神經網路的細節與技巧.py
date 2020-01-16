import keras
train,test=keras.datasets.cifar10.load_data()
def preproc_x(x,flatten=True):
    if flatten:
        x=x.reshape((len(x),-1))
    x=x/255
    return x
def preproc_y(y,num=10):
    y=keras.utils.to_categorical(y,num)
    return y
    
train_x,train_y=train
test_x,test_y=test

train_x=preproc_x(train_x)
test_x=preproc_x(test_x)
train_y=preproc_y(train_y)
test_y=preproc_y(test_y)

from keras.regularizers import l2,l1,l1_l2
#build neurons network
def build_mlp(input_shape,output_units=10,num_neurons=[512,256,128],l2_ratio=0.0001):
    input_layer=keras.layers.Input(input_shape)
    for i,n_units in enumerate(num_neurons):
        if i==0:
            x=keras.layers.Dense(units=n_units,
                                 activation='relu',
                                 kernel_regularizer=l2(l2_ratio),
                                 name='hidden_layer'+str(i+1))(input_layer)
        else:
            x=keras.layers.Dense(units=n_units,
                                 activation='relu',
                                 kernel_regularizer=l2(l2_ratio),
                                 name='hidden_layer'+str(i+1))(x)
        
    out=keras.layers.Dense(units=output_units,
                           activation='softmax',
                           name='output_layer')(x)
    
    model=keras.models.Model(inputs=input_layer,
                             outputs=out)
    
    return model


results={}


for regulizer_ration in [1e-2,1e-4,1e-8,1e-12]:
    keras.backend.clear_session()
    print('Experiment with Regulizer=%.6f'%(regulizer_ration))
    model=build_mlp(input_shape=train_x.shape[1:],
                    l2_ratio=regulizer_ration)
    
    model.summary()
    optimizer=keras.optimizers.SGD(lr=0.001,nesterov=True,momentum=0.95)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    
    model.fit(train_x,train_y,batch_size=256,epochs=50,validation_data=(test_x,test_y))
    
    #collect results
    train_loss=model.history.history['loss']
    valid_loss=model.history.history['val_loss']
    train_acc=model.history.history['acc']
    valid_acc=model.history.history['val_acc']
    
    exp_name_tag='exp-l2-%s'% str(regulizer_ration)
    
    results[exp_name_tag]={'train_loss':train_loss,
                           'valid_loss':valid_loss,
                           'train_acc':train_acc,
                           'valid_acc':valid_acc}
    
    
import matplotlib.pyplot as plt
color_bar=["r", "g", "b", "y", "m", "k"]
plt.figure(figsize=(8,6))

for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train_loss'])),results[cond]['train_loss'], '-', label=cond+'train', color=color_bar[i])
    plt.plot(range(len(results[cond]['valid_loss'])),results[cond]['valid_loss'], '--', label=cond+'val', color=color_bar[i])
plt.title("Loss")
plt.ylim([0, 10])
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train_acc'])),results[cond]['train_acc'], '-', label=cond+'train', color=color_bar[i])
    plt.plot(range(len(results[cond]['valid_acc'])),results[cond]['valid_acc'], '--', label=cond+'val', color=color_bar[i])
plt.title("Accuracy")
plt.legend()
plt.show()