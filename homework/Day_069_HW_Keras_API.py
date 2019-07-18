from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

#主要輸入接收新聞標題本身，即一個整數序列（每個整數編碼一個詞）。
#這些整數在1 到10,000 之間（10,000 個詞的詞彙表），且序列長度為100 個詞
#宣告一個 NAME 去定義Input
main_input = Input(shape=(100,), dtype='int32', name='main_input')


# Embedding 層將輸入序列編碼為一個稠密向量的序列，
# 每個向量維度為 512。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# LSTM 層把向量序列轉換成單個向量，
# 它包含整個序列的上下文信息
lstm_out = LSTM(32)(x)
#插入輔助損失，使得即使在模型主損失很高的情況下，LSTM 層和Embedding 層都能被平穩地訓練
auxiliary_output = Dense(1, activation='sigmoid', name='news_out')(lstm_out)
#輔助輸入數據與LSTM 層的輸出連接起來，輸入到模型
import keras
auxiliary_input = Input(shape=(5,), name='auxiliary_input')
y = keras.layers.concatenate([lstm_out, auxiliary_input])


# 堆疊多個全連接網路層
y = Dense(64, activation='relu')(y)
y = Dense(64, activation='relu')(y)
#作業解答: 新增兩層
y = Dense(64, activation='relu')(y)
y = Dense(32, activation='relu')(y)

# 最後添加主要的邏輯回歸層
main_output = Dense(1, activation='sigmoid', name='main_output')(y)
auxiliary_output = Dense(1 , activation = 'sigmoid' , name = 'auxiliary_output')(y)
# 宣告 MODEL API, 分別採用自行定義的 Input/Output Layer
model = Model(inputs = [main_input , auxiliary_input] ,
              outputs = [main_output , auxiliary_output])
model.compile(optimizer = 'rmsprop',
              loss={'main_output' : 'binary_crossentropy' , 'auxiliary_output' : 'binary_crossentropy'} , 
              loss_weights = {'main_output': 1. , 'auxiliary_output': 0.2})
# 宣告 MODEL API, 分別採用自行定義的 Input/Output Layer
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model.summary()