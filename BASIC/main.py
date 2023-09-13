#%%
print('hello world')
#%%
a=3
b=5
c=a+b
print(c)
#%%
#단열 주석
'''문자열이지만 다열주석으로 사용가능'''
print('나머지연산:',b%a)
'몫연산:', b//a #몫연산 프린트 생략해도 주피터상에서는 출력
#%%
# 데이터셋: array
    # 리스트:[] - 일반배열
    # 튜플:() - 수정 불가능한 리스트
    # 딕셔너리{키:벨류} - 연관배열: 객체의 개념
    # 셋:set()집합

li = [1,2,'3',[4,5]] #리스트 인덱스 숫자로 접근
li[0] = 0
print('0치환이후:',li)
# %%
tu = (1,2,'3',[4,5]) # 튜플 수정 불가
tu[0] = 0
print('0치환이후:',tu)
# %% 
#딕셔너리
di = {'이름':'홍길동',1:19,2:{1,2,3}}
print(di[2])
# %%
# {}블럭이 없어서 들여쓰기가 블럭으로 인식
for l in li:
    print(l)
# %%
for t in tu:
    print(t)
# %%
# dict은 일반 for문으로 출력시 키 값만 빠짐
for d in di:
    print(d)
print('#'*30)
for k,v in di.items():
    print(k,':',v)
# %%
# 구구단 만들기 range를 활용
for i in range(1,10,1):
    print('')
    for j in range(1,10):
        ans = i*j
        if(ans%2==1):
            print(ans,'*',end='\t')
        else:
            print(ans,end='\t')
# %%
# 구구단 만들기 range를 활용
def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans = i*j
            if(ans%2==1):
                print(ans,'*',end='\t')
            else:
                print(ans,end='\t')
# %%
i=5
while(i<10):
    i+=1 #i++없음
    print(i)
# %%
# def 함수명():
def add(a=2,b=4):
    return a+b
print(add(2,3))
print(add())
# %%
#클래스 선언
class person1():
    def __sayHello__(self):#클래스의 메서드이다self
        print('Hello')

anna=person1()
anna.__sayHello__()
# %%
#클래스 선언
class person():
    def __init__(self,name,age):#클래스의 메서드이다self
        self.name=name #클래스 속성
        self.age=str(age)
    def sayHello(self):#클래스의 메서드이다self
        print('Hello '+self.name)
        print("I'm"+self.age+"year old")

anna=person('anna',19)
anna.sayHello()
# %%
