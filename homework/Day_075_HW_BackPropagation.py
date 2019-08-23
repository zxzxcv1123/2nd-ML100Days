import numpy as np
#sigmoid函數可以將任何值都應設到一個位於0~1範圍內的值。將實數轉成為概率值
def nonlin(x,deriv=False): 
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

x=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

#define y for output dataset
y=np.array([[0,0,1,1,]]).T
#seed random numbers to make calculation
#deterministic確定性 (just a good practice)
np.random.seed(1)
#initialize weights randomly with mean 0
#X是(4*3)，為syn0設定(3,3) 輸出後l1為(4*3)
syn0=2*np.random.random((3,3))-1 #"輸入層-第一層隱層"間權重矩陣
#define syn1
#l1為(4,3),syn1設定為(3,1) 這樣l2輸出就會變(4,1)
syn1=2*np.random.random((3,1))-1
iter=0


#神經網路訓練
for iter in range(10000):
    #foreard propagation
    l0=x
    l1=nonlin(np.dot(l0,syn0)) #l0(輸入)*syn0權重=l1的預測輸出(4,3)
    l2=nonlin(np.dot(l1,syn1)) #輸出為(4,1)的矩陣
    
    l1_error=y-l1 #誤差
    l1_delta=l1_error*nonlin(l1,True) #對l1做微分*誤差項
    l2_error=y-l2 
    l2_delta=l2_error*nonlin(l2,True)
    
    #update weights
    syn0+=np.dot(l0.T,l1_delta)#(3*4)*(4*3)
    syn1+=np.dot(l1.T,l2_delta) #(4*3)

    
print("Output After Training:")
print(l1)
print("\n")
print(l2)
