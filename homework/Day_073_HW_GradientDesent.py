cur_x = 3 # The algorithm starts at x=3
lr = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function 

iters_history = [iters]
x_history = [cur_x]

while previous_step_size>precision and iters<max_iters:
    prev_x=cur_x #store current x value in prev_x
    cur_x=cur_x-lr*df(prev_x) #新的x_value
    previous_step_size=abs(cur_x-prev_x) #變動的step值
    iters=iters+1
    print('Iteration',iters,'\nX value is',cur_x)
    #store parameters for plotting
    iters_history.append(iters)
    x_history.append(cur_x)
print('total iterations:',iters)

import matplotlib.pyplot as plt
plt.plot(iters_history,x_history,lw=1.5,color='black')
plt.ylabel('x value',fontsize=16)
plt.xlabel('iters',fontsize=16)

#HW
#使用function y=(x+5)²,learning rate=[0.1,0.0001]
cur_x=3
learning_rate=0.1
precision=0.000001
df=lambda x:2*(x+5)
iters=0
max_iters=1000
step_size=1
#for plot
iters_history=[iters]
x_history=[cur_x]
while step_size>precision and iters<max_iters:
    prev_x=cur_x
    cur_x=cur_x-learning_rate*df(cur_x) #注意是減號
    step_size=abs(cur_x-prev_x)
    iters=iters+1
    #for plot
    x_history.append(cur_x)
    iters_history.append(iters)

#plot
import matplotlib.pyplot as plt
plt.plot(iters_history,x_history,lw=1.5,color='black')
plt.xlabel('iters',fontsize=16)
plt.ylabel('x value',fontsize=16)