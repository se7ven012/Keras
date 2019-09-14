#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.regularizers import l2
from keras.optimizers import SGD,Adam

#%%
#loading datasets
(x_train,y_train),(x_test,y_test) =mnist.load_data()

#x_shape (60000, 28, 28)
print('x_shape',x_train.shape)
#y_shape (60000,)
print('y_shape',y_train.shape)

#x_shape (60000, 28, 28)->(60000,784)
x_train=x_train.reshape(x_train.shape[0],-1)/255.0
x_test=x_test.reshape(x_test.shape[0],-1)/255.0

#换one hot 格式(处理标签数据)
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#network structure 784-10
model = Sequential([
    Dense(units=200,input_dim=784,bias_initializer='one',activation='tanh',kernel_regularizer=l2(0.0003)),
    Dense(units=100,bias_initializer='one',activation='tanh', kernel_regularizer=l2(0.0003)),
    Dense(units=10,bias_initializer='one',activation='softmax', kernel_regularizer=l2(0.0003)),
])

#optimizor
sgd = SGD(lr=0.2)
adam=Adam(lr=0.001)

#计算准确率
model.compile(
    optimizer=sgd,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

#Train a model   epoch是循环周期
model.fit(x_train,y_train,batch_size=32,epochs=10)

loss,accuracy=model.evaluate(x_test,y_test)
print('\ntest loss',loss)
print('test acc:',accuracy)

loss,accuracy=model.evaluate(x_train,y_train)
print('\ntrain loss',loss)
print('train acc:',accuracy)

#%%
