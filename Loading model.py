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
from keras.models import load_model

#%%
#生成100个点
x_data = np.linspace(-0.5,0.5,200)
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#显示随机点
plt.scatter(x_data,y_data)
plt.show()

#%%
#loading model
model = load_model('test.h5')

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

#save weights
model.save_weights('weights.h5')
model.load_weights('weights.h5')
#save network structure, loading network structure
from keras.models import model_from_json
json_string = model.to_json()
model=model_from_json(json_string)

#%%
print(json_string)


#%%
