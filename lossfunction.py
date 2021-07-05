import numpy as np

# 평균제곱오차
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)  # y와 t는 넘파이 배열


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 정답 레이블

y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 2일 확률이 가장 높다고 추정함 (0.6)
print(mean_squared_error(np.array(y1), np.array(t)))

y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]  # 7일 확률이 가장 높다고 추정함 (0.6)
print(mean_squared_error(np.array(y2), np.array(t)))


#교차엔트로피 오차
def cross_entropy_error(y, t):
    delta = 1e-7                          # 아주 작은 값
    return -np.sum(t * np.log(y + delta)) # np.log()에 0을 입력하면 마이너스 무한대를 의미하므로
                                          # 계산이 되지 않는다. 따라서 아주 작은 값을 더해줬습니다.

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]                       # 정답은 2
y = [0.0, 0.05, 0.8, 0.0, 0.05, 0.0, 0.0, 0.1, 0.0, 0.0] # 신경망이 2로 추정

print( cross_entropy_error(np.array(y), np.array(t) ))

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] # 신경망이 7로 추정

print( cross_entropy_error(np.array(y), np.array(t) ))

a = np.array( [1010, 1000, 990] )
# print( np.exp(a) / np.sum(np.exp(a)) ) # 소프트맥스 함수의 계산

c = np.max(a) # c는 입력의 최댓값을 이용합니다.
print( a - c )

print( np.exp(a - c) / np.sum( np.exp(a - c) ))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)
print(np.sum(y))