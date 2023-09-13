#%%
import sqlite3  # sqllite3 데이터베이스
import re       # 정규식
import numpy as np  # 숫자 라이브러리
import pandas as pd # 데이터 처리 라이브러리
import matplotlib as mpl # 그래프 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns #그래프 고도화
# scikit - learn
print("init....")
#%%
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X=data['data']
Y = data['target']
fname = data['feature_names']
tname = data['target_names']
# print(data['DESCR'])
df = pd.DataFrame(X,columns=fname)
df
#%%
#EDA
# 기초통계량
df.describe()
#%%
df.info()
#%%
#Y
#기초시각화
#plt.hist(Y)
plt.plot(df['mean radius'],'.')
#%%
sns.scatterplot(df.iloc[:,:3])
#%%
tdf = df.copy()
tdf['tgt'] = Y
sns.pairplot(tdf.iloc[:,-5:],hue= 'tgt')
#%%
#데이터 전처리
from sklearn.model_selection import train_test_split
X_tr,X_te,Y_tr,Y_te= train_test_split(X,Y,shuffle=True,random_state=1,stratify=Y)
print(X_tr.shape,X_te.shape)
print(Y_tr.shape,Y_te.shape)
plt.hist(Y_tr)
plt.hist(Y_te)
# %%
## 앙상블 모델 Random Forest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RF
def makeRF(i,j):
    rf=RF(max_depth=j,max_leaf_nodes=i)
    rf.fit(X_tr,Y_tr)
    pred = rf.predict(X_te)
    acc = accuracy_score(pred,Y_te)
    print('RF[',i,j,']acc:',acc)
    return acc
accs=[]
beforeACC = 0
bestACC = []
for i in range(2,10):
    for j in range(2,10):
        acc = makeRF(i,j)
        if (acc > beforeACC):
            bestACC = [i,j,acc]
        beforeACC = acc
        accs.append(acc) #리스트에 데이터 추가
#%%
# %%
## 앙상블 모델 Random Forest
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DT
def makeDT(i,j):
    dt=DT(max_depth=j,max_leaf_nodes=i)
    dt.fit(X_tr,Y_tr)
    pred = dt.predict(X_te)
    acc = accuracy_score(pred,Y_te)
    print('DT[',i,j,']acc:',acc)
    return acc
accs=[]
beforeACC = 0
bestACC = []
for i in range(2,10):
    for j in range(2,10):
        acc = makeDT(i,j)
        if (acc > beforeACC):
            bestACC = [i,j,acc]
        beforeACC = acc
        accs.append(acc) #리스트에 데이터 추가
# %%
print(bestACC)
plt.plot(accs)
# %%
from sklearn.ensemble import GradientBoostingClassifier as GB
# 그래디언트 부스팅 - 무겁다
def makeGB(i,j):
    gb=GB(min_samples_split=j, n_estimators=i*50)
    gb.fit(X_tr,Y_tr)
    pred = gb.predict(X_te)
    acc = accuracy_score(pred,Y_te)
    print('GB[',i,j,']acc:',acc)
    return acc
accs=[]
beforeACC = 0
bestACC = []
for i in range(1,10):
    for j in range(2,10):
        acc = makeGB(i,j)
        if (acc > beforeACC):
            bestACC = [i,j,acc]
        beforeACC = acc
        accs.append(acc) #리스트에 데이터 추가
# %%
print(bestACC)
plt.plot(accs)
# %%
from sklearn.ensemble import AdaBoostClassifier as AB
# 그래디언트 부스팅 - 무겁다
def makeAB(i):
    ab=AB(n_estimators=i*20)
    ab.fit(X_tr,Y_tr)
    pred = ab.predict(X_te)
    acc = accuracy_score(pred,Y_te)
    print('AB[',i,']acc:',acc)
    return acc
accs=[]
beforeACC = 0
bestACC = []
for i in range(1,10):
    acc = makeAB(i)
    if (acc > beforeACC):
        bestACC = [i,acc]
    beforeACC = acc
    accs.append(acc) #리스트에 데이터 추가
# %%
print(bestACC)
plt.plot(accs)
# %%
