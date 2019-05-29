from sklearn.model_selection import train_test_split,KFold
import numpy as np
x=np.arange(50).reshape(10,5)# 生成從 0 到 50 的 array，並 reshape 成 (10, 5)
y=np.zeros(10)# 生成一個全零 arrary
y[:5]=1 #將一半的值改為 1
#使用 train_test_split 函數進行切分
train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=0.33,random_state=42)
print(train_X)

#使用 K-fold Cross-validation 來切分資料
kf=KFold(n_splits=5)
i=0
for train_index,test_index in  kf.split(x):
    i+=1
    trian_X,test_X=x[train_index],x[test_index] #matrix先列在行
    train_Y,test_Y=y[train_index],y[test_index]
    print(f'FOLD{i}')
    print(f'X_test{test_X}')
    print(f"Y_test{test_Y}")
    print('-'*30)
    
#作業:假設我們資料中類別的數量並不均衡，在評估準確率時可能會有所偏頗
#試著切分出 y_test 中，0 類別與 1 類別的數量是一樣的
x=np.arange(1000).reshape(200,5)
y=np.zeros(200)
y[:40]=1
#選出Y=0跟1的index
index0=np.where(y==0)[0]
index1=np.where(y==1)[0]
#個別分割
train_X0,test_X0,train_Y0,test_Y0=train_test_split(x[index0],y[index0],test_size=10)
train_X1,test_X1,train_Y1,test_Y1=train_test_split(x[index1],y[index1],test_size=10)
#合併
train_X=np.concatenate([train_X0,train_X1])
test_X=np.concatenate([test_X0,test_X1])
train_Y=np.concatenate([train_Y0,train_Y1])
test_Y=np.concatenate([test_Y0,test_Y1])
