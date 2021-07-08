# 퍼셉트론에서는 활성화 함수로 계단 함수를 이용합니다. 활성화 함수를 계단 함수에서 다른 함수로 변경하는 것이 신경망의 세계로 나아가는 열쇠입니다.
import numpy as np
import matplotlib.pylab as plt
# 계단함수
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

# 넘파이로 계단함수
def step_function(x):
  y = x > 0
  return y.astype(np.int)

# 넘파이를 이용하여 두줄로 계단 함수를 구현할 수 있는 이유
# 넘파이 배열에 부등호 연산을 수행하면 배열의 원소 각각에 부등호 연산을 수행한 bool 배열이 생성됩니다.

# matplotlib 라이브러리를 사용하여 계단 함수를 그래프로 그려봅시다.

def step_function(x):
    return np.array(x > 0, dtype=np.int) # 넘파이 배일여 0보다 클시에 1을 반환, 0보다 작으면 0을 반환합니다.

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축의 범위 지정
plt.show()

# 시그모이드 함수 구현하기

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # np.exp(-x)는 exp(-x) 수식에 해당한다. 인수 x가 넘파이 배열이어도 올바른 결과가 출력됩니다.
    # (넘파이의 브로드캐스트 기능)

#시그모이드 함수를 그래프로 그려봅시다.

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()


# relu 함수
def relu(x):
    return np.maximum(0, x)


# 여기에는 넘파이의 maximum 함수를 사용했습다.
# maximum은 두 입력중 큰 값을 선택해 반환하는 함수입니다.

relu(-0.1)


def identity_function(x):  # 출력층 활성화 함수는 함등함수를 이용했습니다.
    return x


def init_network():  # 가중치와 편향을 초기화하고 이들을 딕셔너리 변수인 network에 저장합니다.
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):  # 입력 신호를 출력으로 변환하는 처리 과정을 구현합니다.
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)  # identity_fuction은 위에서 정의한 출력층 항등 함수를 의미합니다.

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [0.31682708 0.69627909]