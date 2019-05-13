#練習時間¶
#請寫一個函式用來計算 Mean Square Error
import numpy as np
import matplotlib.pyplot as plt
#mse mae
def mean_squared_error(y,yp):
    mse=(sum(y-yp))**2/len(y)
    return mse

def mean_absolute_error(y,yp):
    mae=(sum(np.absolute(y-yp)))/len(y)
    return mae
#創造資料
w = 3
b = 0.5
x_lin = np.linspace(0, 100, 101)
y = (x_lin + np.random.randn(101) * 5) * w + b
plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()
#預測線性模型
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()

MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))