import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
#弱相關
x=np.random.randint(0,50,100)
y=np.random.randint(0,50,100)
np.corrcoef(x,y)
plt.scatter(x,y)
#正相關
y=np.random.randint(0,10,100)+0.5*x
np.corrcoef(x,y)
plt.scatter(x,y)
#負相關
y=np.random.randint(0,10,100)-0.5*x
np.corrcoef(x,y)
plt.scatter(x,y)
