# AND 함수
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2  # 가중치를 곱한 입력의 총합이 임계값을 넘으면 1을 반환하고 그 외에는 0을 반환합니다.
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


AND(0, 0)  # 0을 출력
# AND(1, 0) # 0을 출력
# AND(0, 1) # 0을 출력
# AND(1, 1) # 1을 출력

# 가중치와 편향을 포함한 AND
# 퍼셉트론은 입력 신호에 가중치를 곱한 값과 편향을 합하여, 그 값이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력합니다.
import numpy as np

def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w*x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

AND(0, 0) # 0을 출력


# NAND 게이트와 OR 게이트 구현
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # AND와는 가중치(w와 b)만 다르다.
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# OR(1,0)
NAND(1, 0)


# 퍼셉트론은 XOR 게이트를 구현할 수 없으나 기존 구현한 NAND, OR, AND를 조합해서 구현가능
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


XOR(0, 0)  # 0을 출력
XOR(1, 0)  # 1을 출력
XOR(0, 1)  # 1을 출력
XOR(1, 1)  # 0을 출력