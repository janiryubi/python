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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pycaret.classification import *
cancer = load_breast_cancer()
# %%
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)
df['tgt'] = cancer.target
df
# %%
cres = setup(df,target='tgt',train_size = 0.8,session_id = 0)
cres
# %%kjk
bestmodel = compare_models(sort='Accuracy')
bestmodel
#%%
fmodel = finalize_model(bestmodel)
fmodel
# %%
pred = predict_model(bestmodel,data=df.iloc[:100])
pmean = pred['prediction_score'].mean()
pmean
# %%
# %%
predict_model