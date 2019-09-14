#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.regularizers import l2
from keras.optimizers import SGD,Adam

#%%
#loading datasets
(x_train,y_train),(x_test,y_test) =mnist.load_data()

#x_shape (60000, 28, 28)
print('x_shape',x_train.shape)
#y_shape (60000,)
print('y_shape',y_train.shape)

#x_shape (60000, 28, 28)->(60000,28,28,1)
x_train=x_train.reshape(-1,28,28,1)/255.0
x_test=x_test.reshape(-1,28,28,1)/255.0

#换one hot 格式(处理标签数据)
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#model
model=Sequential()

#第一个卷积层
model.add(Convolution2D(
    input_shape=(28,28,1),#输入平面
    filters=32,#卷积核/滤波器个数
    kernel_size=5,#卷积窗口大小
    strides=1,#步长
    padding='same',#padding方式 same/valid
    activation = 'relu'
))
#第一个池化层
model.add(MaxPooling2D(
    pool_size=2,#池化面积
    strides=2,#步长
    padding='same',#padding方式 same/valid
))

#第二个卷积核
model.add(Convolution2D(64,5,strides=1,padding='same',activation='relu'))
#第二个池化层
model.add(MaxPooling2D(2,2,'same'))

#将第二个池化层输出扁平化为1维
model.add(Flatten())

#第一个全联接层
model.add(Dense(1024,activation='relu'))
#dropout
model.add(Dropout(0.5))
#第二个全联阶层
model.add(Dense(10,activation='softmax'))

#optimizor
adam=Adam(0.001)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=10)

loss,accuracy= model.evaluate(x_test,y_test)

print('test loss:',loss)
print('test acc:',accuracy)



#%%
