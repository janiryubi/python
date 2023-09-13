# 모듈속성
mp1='모듈프로퍼티'
# 모듈내 함수
def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans = i*j
            if(ans%2==1):
                print(ans,'*',end='\t')
            else:
                print(ans,end='\t')


class person():
    def __init__(self,name,age):#클래스의 메서드이다self
        self.name=name #클래스 속성
        self.age=str(age)
    def sayHello(self):#클래스의 메서드이다self
        print('Hello '+self.name)
        print("I'm"+self.age+"year old")