import math
import numpy as np
import matplotlib.pyplot as plt

class Calculator:
    """
    간단한 계산기 클래스로 수학 식을 평가할 수 있습니다.
    기본 산술 연산을 지원하며 샌팅야드 알고리즘을 따릅니다.
    """

    def __init__(self):
        """
        빈 연산자 스택과 출력 큐로 계산기를 초기화합니다.
        """
        self.operator_stack = []  # 연산자를 저장하는 스택
        self.output_queue = []    # 후위 표기법으로 변환된 수식을 저장하는 큐

    def is_operator(self, token):
        """
        주어진 토큰이 연산자인지 확인합니다. (+, -, *, /).

        매개변수:
            token (str): 확인할 토큰.

        반환값:
            bool: 토큰이 연산자이면 True, 그렇지 않으면 False.
        """
        return token in {'+', '-', '*', '/'}  # 주어진 토큰이 연산자인지 확인하는 함수

    def get_precedence(self, operator):
        """
        주어진 연산자의 우선순위 수준을 반환합니다.

        매개변수:
            operator (str): 우선순위를 얻을 연산자.

        반환값:
            int: 우선순위 수준입니다. 높은 값은 높은 우선순위를 나타냅니다.
        """
        if operator in {'+', '-'}:
            return 1
        elif operator in {'*', '/'}:
            return 2
        return 0  # 연산자의 우선순위를 반환하는 함수

    def shunting_yard(self, expression):
        """
        샌팅야드 알고리즘을 사용하여 중위 표현식을 후위로 변환합니다.

        매개변수:
            expression (str): 중위 수학 표현식.
        """
        for token in expression:
            if token.isdigit():
                self.output_queue.append(float(token))  # 피연산자는 바로 출력 큐에 추가
            elif self.is_operator(token):
                # 스택에 있는 연산자의 우선순위가 높거나 같으면 출력 큐에 추가
                while (self.operator_stack and
                       self.is_operator(self.operator_stack[-1]) and
                       self.get_precedence(self.operator_stack[-1]) >= self.get_precedence(token)):
                    self.output_queue.append(self.operator_stack.pop())
                self.operator_stack.append(token)  # 현재 연산자를 스택에 추가
            elif token == '(':
                self.operator_stack.append(token)  # 왼쪽 괄호는 스택에 추가
            elif token == ')':
                # 오른쪽 괄호를 만날 때까지 스택의 연산자를 출력 큐에 추가
                while self.operator_stack and self.operator_stack[-1] != '(':
                    self.output_queue.append(self.operator_stack.pop())
                if not self.operator_stack or self.operator_stack[-1] != '(':
                    raise ValueError("맞지 않는 괄호.")
                self.operator_stack.pop()  # 왼쪽 괄호는 스택에서 제거

        # 스택에 남아 있는 모든 연산자를 출력 큐에 추가
        while self.operator_stack:
            if self.operator_stack[-1] == '(':
                raise ValueError("맞지 않는 괄호.")
            self.output_queue.append(self.operator_stack.pop())

    def calculate_postfix(self):
        """
        후위 표현식의 결과를 계산합니다.

        반환값:
            float: 후위 표현식의 결과.
        """
        result_stack = []  # 계산을 위한 스택
        for token in self.output_queue:
            if isinstance(token, float):
                result_stack.append(token)  # 피연산자는 스택에 추가
            elif self.is_operator(token):
                # 연산자를 만나면 스택에서 필요한 피연산자를 꺼내 계산 후 결과를 스택에 추가
                if len(result_stack) < 2:
                    raise ValueError("연산자 {}에 대한 피연산자가 충분하지 않습니다.".format(token))
                operand2 = result_stack.pop()
                operand1 = result_stack.pop()
                if token == '+':
                    result_stack.append(operand1 + operand2)
                elif token == '-':
                    result_stack.append(operand1 - operand2)
                elif token == '*':
                    result_stack.append(operand1 * operand2)
                elif token == '/':
                    if operand2 == 0:
                        raise ValueError("0으로 나눌 수 없습니다.")
                    result_stack.append(operand1 / operand2)
        if len(result_stack) != 1:
            raise ValueError("유효하지 않은 표현식.")
        return result_stack[0]  # 최종 결과를 반환

    def evaluate_expression(self, expression):
        """
        주어진 중위 수학 표현식을 평가합니다.

        매개변수:
            expression (str): 중위 수학 표현식.

        반환값:
            float: 표현식의 결과.
        """
        self.operator_stack = []  # 연산자 스택 초기화
        self.output_queue = []    # 출력 큐 초기화

        self.shunting_yard(expression)  # 중위 표기법을 후위 표기법으로 변환
        result = self.calculate_postfix()  # 후위 표기법을 계산하여 결과 획득

        return result

# 예제 사용
calculator = Calculator()

# 수식 입력 받기
expression = input("수식 입력 (예: ((3 + 5) * (2 - 7)): ")
try:
    result = calculator.evaluate_expression(expression)
    print("Result:", result)
except ValueError as e:
    print("Error:", str(e))

############################################################3

class EngineeringCalculator(Calculator):
    """
    공학용 계산기 클래스로 삼각함수, 상수, 팩토리얼, 행렬 연산 및 사용자 정의 함수 그래프 그리기 기능을 제공합니다.
    Calculator 클래스를 상속받아 확장되었습니다.
    """

    def __init__(self):
        """
        Calculator 클래스의 생성자를 호출하여 초기화합니다.
        """
        super().__init__()

    def sin(self, angle):
        """
        주어진 각도의 사인 값을 계산합니다.

        매개변수:
            angle (float): 각도 (도수법).

        반환값:
            float: 사인 값.
        """
        return math.sin(math.radians(angle))

    def cos(self, angle):
        """
        주어진 각도의 코사인 값을 계산합니다.

        매개변수:
            angle (float): 각도 (도수법).

        반환값:
            float: 코사인 값.
        """
        return math.cos(math.radians(angle))

    def tan(self, angle):
        """
        주어진 각도의 탄젠트 값을 계산합니다.

        매개변수:
            angle (float): 각도 (도수법).

        반환값:
            float: 탄젠트 값.
        """
        return math.tan(math.radians(angle))

    def pi(self):
        """
        원주율(파이) 값을 반환합니다.

        반환값:
            float: 원주율 값.
        """
        return math.pi

    def e(self):
        """
        자연상수(e) 값을 반환합니다.

        반환값:
            float: 자연상수 값.
        """
        return math.e

    def factorial(self, n):
        """
        주어진 정수 n의 팩토리얼 값을 계산합니다.

        매개변수:
            n (int): 계산할 팩토리얼의 정수 값.

        반환값:
            int: n의 팩토리얼 값.
        """
        if n == 0 or n == 1:
            return 1
        else:
            return n * self.factorial(n - 1)

    def matrix_addition(self, matrix1, matrix2):
        """
        두 행렬의 덧셈을 계산합니다.

        매개변수:
            matrix1 (numpy.ndarray): 첫 번째 행렬.
            matrix2 (numpy.ndarray): 두 번째 행렬.

        반환값:
            numpy.ndarray: 두 행렬의 덧셈 결과.
        """
        return np.add(matrix1, matrix2)

    def matrix_subtraction(self, matrix1, matrix2):
        """
        두 행렬의 뺄셈을 계산합니다.

        매개변수:
            matrix1 (numpy.ndarray): 첫 번째 행렬.
            matrix2 (numpy.ndarray): 두 번째 행렬.

        반환값:
            numpy.ndarray: 두 행렬의 뺄셈 결과.
        """
        return np.subtract(matrix1, matrix2)

    def matrix_multiplication(self, matrix1, matrix2):
        """
        두 행렬의 곱셈을 계산합니다.

        매개변수:
            matrix1 (numpy.ndarray): 첫 번째 행렬.
            matrix2 (numpy.ndarray): 두 번째 행렬.

        반환값:
            numpy.ndarray: 두 행렬의 곱셈 결과.
        """
        return np.dot(matrix1, matrix2)

    def plot_user_defined_function(self):
        """
        사용자로부터 입력 받은 함수식과 x 값의 범위를 이용하여 함수 그래프를 출력합니다.
        """
        # 함수식 입력 받기
        expression = input("함수식을 입력하세요 (예: x**2 - 2*x + 1): ")
        
        # 입력받은 함수식을 파이썬 함수로 변환
        def user_defined_function(x):
            return eval(expression)
        
        # x 값의 범위 입력 받기
        x_start = float(input("x 값의 시작 범위를 입력하세요: "))
        x_end = float(input("x 값의 종료 범위를 입력하세요: "))
        
        # 함수 그래프 그리기
        x_values = np.linspace(x_start, x_end, 100)
        y_values = user_defined_function(x_values)
        plt.plot(x_values, y_values, label=expression)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

# 예제 사용
eng_calculator = EngineeringCalculator()

# 각도 입력 받기
angle = float(input("삼각함수의 각도 입력 (도수법): "))
try:
    result_sin = eng_calculator.sin(angle)
    result_cos = eng_calculator.cos(angle)
    result_tan = eng_calculator.tan(angle)

    print("sin({}°): {}".format(angle, result_sin))
    print("cos({}°): {}".format(angle, result_cos))
    print("tan({}°): {}".format(angle, result_tan))

    # 상수 사용 예제
    result_pi = eng_calculator.pi()
    result_e = eng_calculator.e()

    print("pi: {}".format(result_pi))
    print("e: {}".format(result_e))
except ValueError as e:
    print("Error:", str(e))


# 예제 사용
eng_calculator2 = EngineeringCalculator()

# 팩토리얼 계산 예제
n = int(input("팩토리얼 계산할 정수 입력: "))
try:
    result_factorial = eng_calculator2.factorial(n)
    print("{}! = {}".format(n, result_factorial))
except ValueError as e:
    print("Error:", str(e))

# 예제 사용
eng_calculator3 = EngineeringCalculator()

# NumPy를 사용한 행렬 연산
matrix1 = np.array(eval(input("행렬1 입력 (예: [[1, 2], [3, 4]]): ")))
matrix2 = np.array(eval(input("행렬2 입력 (예: [[5, 6], [7, 8]]): ")))

try:
    result_matrix_addition = eng_calculator3.matrix_addition(matrix1, matrix2)
    result_matrix_subtraction = eng_calculator3.matrix_subtraction(matrix1, matrix2)
    result_matrix_multiplication = eng_calculator3.matrix_multiplication(matrix1, matrix2)

    print("행렬 덧셈:\n", result_matrix_addition)
    print("행렬 뺄셈:\n", result_matrix_subtraction)
    print("행렬 곱셈:\n", result_matrix_multiplication)
except ValueError as e:
    print("Error:", str(e))


# 예제 사용
eng_calculator4 = EngineeringCalculator()
eng_calculator4.plot_user_defined_function()


     