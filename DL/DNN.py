#%% tensorflow
# * 케라스 python < 파이토치 JAVA < 텐서플로우 c
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
# %%
print(tf.__version__)
# %%
# 데이터가져오기
mnist = tf.keras.datasets.mnist
#dir(mnist.load_data())
(X_tr,Y_tr),(X_te,Y_te) = (mnist.load_data())
print(X_tr.shape,X_te.shape)
print(Y_tr.shape,Y_te.shape)
# %%
for i in range(10):
    plt.imshow(X_tr[i])
    plt.show()
    print(Y_tr[i])
#%%
print('최대:',X_tr[0].max(),'최소:',X_tr[0].min())
#%%
# minmax 처리(전처리)
(x_tr,y_tr) = (X_tr/255,Y_tr)
(x_te,y_te) = (X_te/255,Y_te)
print(x_tr[0].max())
plt.hist(y_tr)
#%%
# 모델결정 ANN
layer=[
    tf.keras.layers.Flatten(input_shape=(28,28)), #펼친다
    tf.keras.layers.Dense(10,activation = 'softmax') # unit
]
model = tf.keras.models.Sequential(layer)
model.summary()
#%%
# 최적화함수 결정 optimizer = 
# 손실(에러) 결정 loss = 
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
#%%
# 학습하기
model.fit(x_tr,y_tr,epochs = 10)
# %%
