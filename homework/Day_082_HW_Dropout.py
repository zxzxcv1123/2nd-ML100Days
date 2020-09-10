import keras 

(x_train, y_train),(x_test, y_test)=keras.datasets.cifar10.load_data()

def preproc_x(x, flatten=True):
    x = x / 255.
    if flatten:
        x = x.reshape((len(x), -1))
    return x

def preproc_y(y, num_classes=10):
    if y.shape[-1] == 1:
        y = keras.utils.to_categorical(y, num_classes)
    return y

#x_train, y_train = train
#x_test, y_test = test

# Preproc the inputs
x_train = preproc_x(x_train)
x_test = preproc_x(x_test)

# Preprc the outputs
y_train = preproc_y(y_train)
y_test = preproc_y(y_test)


from keras.layers import Dropout

"""
建立神經網路，並加入 dropout layer
"""
def build_mlp(input_shape, output_units=10, num_neurons=[512, 256, 128,128], drp_ratio=0.4):
    input_layer = keras.layers.Input(input_shape)
    
    for i, n_units in enumerate(num_neurons):
        if i == 0:
            x = keras.layers.Dense(units=n_units, 
                                   activation="relu", 
                                   name="hidden_layer"+str(i+1))(input_layer)
            x = Dropout(drp_ratio)(x)
        else:
            x = keras.layers.Dense(units=n_units, 
                                   activation="relu", 
                                   name="hidden_layer"+str(i+1))(x)
            x = Dropout(drp_ratio)(x)
    
    out = keras.layers.Dense(units=output_units, activation="softmax", name="output")(x)
    
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    return model

"""
Code Here
設定超參數
"""
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 256
MOMENTUM = 0.95
Dropout_EXP = 0.25

results = {}
"""
Code Here
撰寫你的訓練流程並將結果用 dictionary 紀錄
"""
model=build_mlp(input_shape=x_train.shape[1:],drp_ratio=Dropout_EXP)
model.summary()
optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=x_train,
          y=y_train,
          batch_size=256,
          epochs=50,
          verbose=-1,
          validation_data=(x_test,y_test),
          shuffle=True)

#collect results
train_loss=model.history.history['loss'] #list
valid_loss=model.history.history['val_loss']
train_acc =model.history.history["accuracy"]
valid_acc = model.history.history["val_accuracy"]

"""Code Here
將結果繪出
"""
import matplotlib.pyplot as plt
#loss
plt.plot(range(len(train_loss)),train_loss,label='train loss')
plt.plot(range(50),valid_loss,label="valid loss")
plt.legend()
plt.title('Loss')
plt.show()
#accuracy
plt.plot(range(len(train_acc)), train_acc, label="train accuracy")
plt.plot(range(len(valid_acc)), valid_acc, label="valid accuracy")
plt.legend()
plt.title("Accuracy")
plt.show()

