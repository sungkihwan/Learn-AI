import numpy as np

# 기울기 구현하기
def numerical_gradient(f, x):  # 함수와 넘파이 배열 x 입력
    # 넘파이 배열 x의 각 원소에 대해서 수치 미분을 구합니다.
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같고 그 원소가 모두 0인 배열을 생성

    for idx in range(x.size):  # x의 요소수만큼 반복
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 중심차분
        x[idx] = tmp_val  # 값 복원
    return grad


import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h) 계산
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 값 복원

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x ** 2)
    else:
        return np.sum(x ** 2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()


# 경사하강법 구현

def gradient_descent(f, init_x, lr, step_num=100):
    # f는 최적화하려는 함수, init_x는 초깃값, lr은 학습률, step_num은 반복 횟수를 의미합니다.

    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)  # 함수의 기울기 구하기
        x -= lr * grad

    return x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


learningrate = 0.1

init_x = np.array([-3.0, 4.0])

gradient_descent(function_2, init_x, learningrate, step_num=100)

import numpy as np
import matplotlib.pylab as plt

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() ) # 시각화를 위해 x_history에 x의 기록을 입력해줍니다.

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)) # 값이 너무 큰 값으로 발산합니다.

# 학습률이 너무 작은 예 : lr = 1e-10
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100) # 거의 학습되지 않은 채 종료합니다.


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 평균 0, 표준편차 1인 가우시안 정규분포 난수를 2X3 배열 생성

    def predict(self, x):  # 예측 수행
        return np.dot(x, self.W)  # x와 self.W 내적

    def loss(self, x, t):  # x는 입력, t는 정답 레이블
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)  # 교차 엔트로피 오차 이용

        return loss


net = simpleNet()
print(net.W)  # 가중치 매개변수 # 평균 0, 표준편차 1 정규분포 난수 생성

x = np.array([0.6, 0.9])

p = net.predict(x)
print(p)  # 소프트 맥스함수를 거치지 않아 확률로 나오지 않았습니다.
# 어차피 최댓값의 인덱스를 구해야 하므로 상관없습니다.

print(np.argmax(p))  # 최댓값의 인덱스

t = np.array([0, 0, 1])  # 정답 레이블
print(net.loss(x, t))  # 손실 함수 구하기


def f(W):  # net.W를 인수로 받아 손실 함수를 계산하는 새로운 함수 정의
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)  # 손실 함수의 기울기

print(dW)
# dW가 0.2의 의미는 w을 h만큼 늘리면 손실 함수의 값은 0.2h만큼 증가합니다.
# 손실 함수를 줄인다는 관점으로 -0.54는 양의 방향으로 갱신하고
# 0.2는 음의 방향으로 갱신해줘야 합니다.