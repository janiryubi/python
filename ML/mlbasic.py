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
# %%
from sklearn import datasets as data
#print(data)
#dir(data)
iris = data.load_iris()
# 전처리
irdata = iris.data # 데이터
irtgt = iris.target # 라벨링
feature = iris.feature_names #데이터 컬럼명
tgtname = iris.target_names # 
#irdata
#type(iris)
# %%
feature = ['sl','sw','pl','pw']
# %%
df = pd.DataFrame(irdata, columns=feature)
df
# %%
# df.plot(kind= "kde") #기초그래프 
# kind종류 -> bar hist pie area scatter box 등
df.plot(style= ".")
# %%
df.describe() # 기초통계량 요약
# 중위수와 평균을 보고 얼만큼 퍼져있는지 확인(정규분포와 비정규분포)
# %%
df.info() # 데이터타입 요약
# %%
tgtname
# %%
# 카테고리별 갯수 히스토그램
plt.hist(irtgt)
# plt.plot(irtgt,'.') # target 값 .으로 나타내기
# %%
df['tgt'] = irtgt
df
# %%
sns.pairplot(df,hue='tgt',palette = "pastel") # tgt 기반으로 색깔로 나누겠다
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #정확도 계산하기
X_tr,X_te,Y_tr, Y_te= train_test_split(irdata,irtgt,test_size=0.3,shuffle=True,random_state=1)
print(X_tr.shape,X_te.shape)
print(Y_tr.shape,Y_te.shape)
# %%
# KNN
# 하이퍼 파라미터 튜닝
for i in range(3,20,2): #range(시작,종료,스텝)
    print('KNN:',i)
    knn3 = KNeighborsClassifier(n_neighbors=i) # 모델지정하기
    knn3.fit(X_tr,Y_tr) # 학습시키기
    pred = knn3.predict(X_te) # 시험보기
    print(pred) # 시험본 답 출력
    print(Y_te) # 실제 답
    acc = accuracy_score(pred,Y_te)
    print("점수,[",i,"]:",acc)
# %% 나 혼자 해본
# df.duplicated().sum()
# df.loc[df.duplicated(),:]
# %%
# SVC
from sklearn.svm import SVC
for i in range(1,10): #range(시작,종료,스텝)
    svc = SVC(C=i)
    svc.fit(X_tr,Y_tr)
    pred = svc.predict(X_te)
    print(pred)
    print(Y_te)
    acc = accuracy_score(pred,Y_te)
    print('SVM[',i,']acc:',acc)
# %%
# 디시전트리
from sklearn.tree import DecisionTreeClassifier
for j in range(2,10): # 가지의 개수
    for i in range(2,10): # 깊이
        dt = DecisionTreeClassifier(max_depth=i,min_samples_leaf=j)
        dt.fit(X_tr,Y_tr)
        pred = dt.predict(X_te)
        print(pred)
        print(Y_te)
        acc = accuracy_score(pred,Y_te)
        print('Desicion Tree[',i,']acc:',acc)
