#%%
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.layers.recurrent import LSTM,SimpleRNN,GRU
from keras.regularizers import l2
from keras.optimizers import SGD,Adam

#%%
#序列长度-一共28行
time_steps=28
#数据长度-一行的28个像素
input_size=28
#hidden layer cell number
cell_size = 50

#loading data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#(60000,784)-->(60000,28,28)#序列长度，数据长度
x_train=x_train/255.0
x_test=x_test/255.0

#labels convert to one hot
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#setup model
model=Sequential()

#RNN
model.add(SimpleRNN(
    units = cell_size, #output
    input_shape = (time_steps,input_size), #input
))

#outputlayer
model.add(Dense(10,activation='softmax'))

#optimizor
adam=Adam(0.001)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=64,epochs=10)

loss,accuracy = model.evaluate(x_test,y_test)

print('test loss:',loss)
print('test acc:',accuracy)

#save model
model.save('model.h5')

#%%
