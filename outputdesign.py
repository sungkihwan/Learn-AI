# 일반적으로 회귀에는 항등 함수를, 분류에는 소프트맥스 함수를 사용합니다.
import numpy as np
import matplotlib.pylab as plt


def identity_function(x):  # 출력층 활성화 함수는 함등함수를 이용했습니다.
    return x


def softmax(a):
    exp_a = np.exe(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# 오버플로우 방지
# 소프트맥스 함수 출력의 총합은 1 입니다.

#  현업에서 출력층의 소프트맥스 함수는 생략하는 것이 일반적입니다.
#  지수 함수 계산에 드는 자원 낭비를 줄이고자 하는 것이 이유입니다.
#  신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식합니다.
#  그리고 소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않습니다.
#  결과적으로 신경망으로 분류할 때는 출력층의 소프트맥스 함수를 생략해도 됩니다.

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
