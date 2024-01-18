# test-01-18

math 라이브러리: 수학 함수 및 상수를 사용하기 위해 math 라이브러리를 import했습니다.
NumPy: 수학적인 배열 및 행렬 연산을 위해 NumPy를 import했습니다.
Matplotlib: 그래프를 그리기 위해 Matplotlib를 import했습니다.
클래스 및 메서드:

Calculator 클래스:
__init__(self): 계산기의 초기 상태를 설정합니다. 빈 연산자 스택과 출력 큐를 생성합니다.
is_operator(self, token): 주어진 토큰이 연산자인지 확인합니다.
get_precedence(self, operator): 주어진 연산자의 우선순위를 반환합니다.
shunting_yard(self, expression): 샌팅야드 알고리즘을 사용하여 중위 표현식을 후위 표현식으로 변환합니다.


self.operator_stack 및 self.output_queue는 클래스의 속성으로, 각각 연산자를 저장하는 스택과 후위 표기법으로 변환된 수식을 저장하는 큐입니다.
is_operator 메서드는 주어진 토큰이 연산자인지 확인합니다.
get_precedence 메서드는 주어진 연산자의 우선순위를 반환합니다.
shunting_yard 메서드는 샌팅야드 알고리즘을 사용하여 중위 표현식을 후위 표현식으로 변환합니다. 피연산자는 출력 큐에 추가하고, 연산자는 스택에 추가 또는 출력 큐에 추가됩니다. 괄호 처리 및 우선순위 비교가 이루어집니다.

for token in expression:: 입력된 중위 표현식을 한 토큰씩 반복합니다.
