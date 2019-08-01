import numpy as np
import matplotlib.pyplot as plt

#Sigmoid
#sigmpid=lambda x:1/(1+np.exp(-x))
def sigmoid(x):
    return(1/(1+np.exp(-x)))
##sigmoid 維分
#def sigmoid(x):
#    return(x*(1-x))

#linespace generate an array from start and stop value
#with requested number of elements.Example 10 elements or 100 elements
x=np.linspace(-10,10,100)
#prepare the plot,associate the color r(ed) or b(lue) and the label
plt.plot(x,sigmoid(x),'b',label='linspace(-10~10,10)')
plt.grid()
plt.title('sigmoid Function')
plt.text(4,0.8,r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=15)
plt.show()

#softmax
def softmax(x):
    return np.exp(x)/float(sum(np.exp(x)))
#x=np.arange(0,1,0.01)
x=np.linspace(-5,5,100)
#列印所有Softmax 值並輸出成一陣列
print(softmax(x))
plt.plot(x, softmax(x), 'r')
plt.show()


#Relu 
#Rectified Linear Unit  (0 if x<0, or x)
def ReLU(x):
    return abs(x)*(x>0)
def dReLU(x):
    return (1*(x>0))
#linespace generate an array from start and stop value
#with requested number of elements
x=np.linspace(-10,10,100)
plt.plot(x,ReLU(x),'r')
plt.plot(x,dReLU(x),'b')
plt.legend(['ReLU','dReLU'],loc='best')
plt.show()