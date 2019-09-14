#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
#傻瓜式网络结构模块
from keras.models import Sequential
#Dense全连接层
from keras.layers import Dense

#%%
#生成100个点
x_data = np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data*0.2 + 0.7 + noise

#显示随机点
plt.scatter(x_data,y_data)
plt.show()

#%%
#构建一个顺序模型
model = Sequential()

#添加全连接层
model.add(Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')
#sgd = 随即梯度下降
#mes = 均方误差

#
for step in range(5001):
    #每训练一次
    cost=model.train_on_batch(x_data,y_data)
    if step % 500 == 0:
        print('cost:',cost)

#print wight, bias
W,b = model.layers[0].get_weights()
print('W:',W,'b:',b)

#predict y_pred
y_pred = model.predict(x_data)

#display random dots
plt.scatter(x_data,y_data)
#display prediction result
plt.plot(x_data,y_pred)
plt.show()


