#%%
#모듈화
#직접 불러서 네이밍
import prs.prs99 as pp
pp.make99()
# %%
# 폴더 안에서 파일을 불러서 네이밍
from prs import prs99 as p9
p9.make99()
# %%
from prs.prs99 import *
make99()
person ()
# %%
import main as m 
m.make99