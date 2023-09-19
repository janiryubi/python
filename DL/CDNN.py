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
cifa10 = tf.keras.datasets.cifar10
cifac = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']
#%%
#dir(cifa10.load_data())
(X_tr,Y_tr),(X_te,Y_te) = (cifa10.load_data())
print(X_tr.shape,X_te.shape)
print(Y_tr.shape,Y_te.shape)
# dir(cifa10)
Y_tr[0][0]
# %%
for i in range(10):
    plt.imshow(X_tr[i])
    plt.show()
    print(cifac[Y_tr[i][0]])
#%%
print('최대:',X_tr[0].max(),'최소:',X_tr[0].min())
#%%
# minmax 처리(전처리)
(x_tr,x_te) = (X_tr/255,X_te/255)
# one-hot encoding 형태로 변환
y_tr = tf.keras.utils.to_categorical(Y_tr.reshape(-1))
y_te = tf.keras.utils.to_categorical(Y_te.reshape(-1))
print('최대값',x_tr[0].max())
plt.hist(y_tr)
#%%
print(y_tr[0],Y_tr[0])

#%%
# 모델결정 DNN
layer=[
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(1024,activation = 'relu'), #펼친다
    tf.keras.layers.Dense(128,activation = 'relu'), # 얼마나 영향?
    tf.keras.layers.Dense(64,activation = 'relu'),
    tf.keras.layers.Dense(10,activation = 'softmax') # unit
]
model = tf.keras.models.Sequential(layer)
model.summary()
#%%
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                  min_delta=0.001,
                                  patience=10,
                                  verbose=1,
                                  mode='auto',
                                  baseline=None,
                                  restore_best_weights=False)
#%%
# 최적화함수 결정 optimizer = 
# 손실(에러) 결정 loss = 
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy', # 여기 변경됨
              metrics = ['accuracy'])
#%%
# 학습하기
hist = model.fit(x_tr,y_tr,
                 epochs = 100,
                 batch_size=1000,
                 verbose=1,
                 callbacks=[es])
# %%
# 시험보기
loss,acc = model.evaluate(x_te,y_te)
# %%
# 차이가 많이나면 과적합이 나고 있는 것, 공부는 잘하지만 시험은 못봄
print(loss,acc)
# %%
