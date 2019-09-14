#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
#傻瓜式网络结构模块
from keras.models import Sequential
#Dense全连接层
from keras.layers import Dense,Activation
#
from keras.optimizers import SGD

#%%
#生成100个点
x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#显示随机点
plt.scatter(x_data,y_data)
plt.show()

#%%
#构建一个顺序模型
model = Sequential()

#1-10-1网络结构

#添加全连接层
model.add(Dense(units=10,input_dim=1,activation='tanh'))#输出 和 输入
#激活函数可以解决非线性问题
# model.add(Activation('tanh'))
#输出层
model.add(Dense(units=1,activation='tanh'))#输出
# model.add(Activation('tanh'))

#自定义优化算法
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd,loss='mse')
#sgd = 随即梯度下降
#mes = 均方误差

#
for step in range(3001):
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

#save model
model.save('test.h5')


#%%
