*python설치
*가상환경 생성
    - pip install virtualenv
    - python.exe. -m pip install --upgrade pip
    - virtualenv venv
    or python -m venv venv
*보안정책 개방
    - Set-ExecutionPolicy RemoteSigned(Powershall 관리자 권한 실행)
*가상환경 활성화(venv)
    - .\venv\Scripts\activate
*가상환경 비활성화
    - deactivate
*스크립트언어의 장점을 살려 블록단위 개발을 위한 환경: jupyter 환경 Anaconda
    - 연구목적의 개발에 큰 장점
    - 파이썬을 응용한 다양한 프로그램 개발로 발전 : 개발환경에서 주피터 역할 추가
*특징
    - 스크립트언어(컴파일 과정이 없음)
    - 현자유도 중간정도 JAVA(형변환이 필요) < python(문자열과 숫자가 결합은 안되지만 변수 대입시 묵시적 변환)> php(문자열과 숫자의 결합이 자유로움)