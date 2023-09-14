#%%
import sqlite3  # sqllite3 데이터베이스
import re       # 정규식
import numpy as np  # 숫자 라이브러리
import pandas as pd # 데이터 처리 라이브러리
import matplotlib as mpl
import matplotlib.pyplot as plt # 그래프
import seaborn as sns #그래프 고도화
# %%
from konlpy.tag import Kkma
# %%
kkma=Kkma()
# %%
res = kkma.pos('안녕하세요 여러분 만나서 반갑습니다.')
res
# %%
def get_Pos(txt='안녕하세요 여러분 만나서 반갑습니다.'):
    res = kkma.pos(txt)
    reqPos = ['NNG','NNP','NP','VV','VA','VCN','JC','MAC','EFA','EFQ','EFO','EFR','EFI'] #'EFN' 'VCP'
    wset=[]
    for r in res:
        if(r[1] in reqPos):
            wset.append(r[0])
            # print(r)
    return(' '.join(wset))
get_Pos()
# %%
fname = '..\src\현진건-운수_좋은_날+B3356-개벽.txt'
with open(fname,encoding='utf8') as f:
    r=f.readlines()
print(r)
# %%
r[:10]
# \n없애구
# %%
#,한줄로
lucky = ''.join(r)
lucky[:100]
# %%
lucky = lucky.replace('\n\n','{nn}')
lucky = lucky.replace('\n','')
lucky = lucky.replace('{nn}','.')
print(lucky)
# %%
lucks = lucky.split('.')
lucks[:10]

# %%
copus=[]
for luck in lucks:
    ltxt = get_Pos(luck)
    copus.append(ltxt)
copus[:10]

# %%
#t = '새침하게 흐린 품이 눈이 올 듯하더니 눈은 아니 오고 얼다가 만 비가 추적추적 내리는 날이었다'
#get_Pos(t)
# %%
# 행렬의 많이 나오는 단어 없애고, 단어별 유사도 알아내기
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#단어행렬 CBOW
cvect = CountVectorizer()
cvfit = cvect.fit_transform(copus)
cvtable = cvfit.toarray()
print(cvtable.shape)
print(cvect.vocabulary_)
# %%
plt.imshow(cvtable[:100,:100])

# %%
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#빈도-역빈도 TF-IDF
tvect = TfidfVectorizer()
tvfit = tvect.fit_transform(copus)
tvtable = tvfit.toarray()
print(tvtable.shape)
print(tvect.vocabulary_)
# %%
plt.imshow(tvtable[:100,:100])
# %%
cdf = pd.DataFrame(cvtable[:100,:100])
cdf
# %%
tdf = pd.DataFrame(tvtable[:100,:100])
tdf
# %%
