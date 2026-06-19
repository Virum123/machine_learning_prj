import pandas as pd
from sklearn.datasets import load_iris # 데이터셋
from sklearn.model_selection import train_test_split # 데이터셋 분리

from sklearn.naive_bayes import GaussianNB

#  나이브 베이즈 ==>
from sklearn.naive_bayes import GaussianNB # 데이터 특징 가우시안 정규분포를 취할 떄
# 분류 모델로 사용 ==> GaussianNB
from sklearn import metrics # 혼동행렬
from sklearn.metrics import accuracy_score

dataset = load_iris()
print(dataset)

# 모델에 학습 시킬 데이터셋 준비
# train/test 분리
train_x, test_x, train_y, test_y = \
    train_test_split(dataset['data'], dataset['target'], test_size=0.2, random_state=42) 

print(train_x[:5])
print(test_x[:5])

# 가우시안 나이브베이즈 모델준비
gnb_model = GaussianNB()

# 모델 학습
gnb_model.fit(train_x, train_y)

# 모델 예측
pred = gnb_model.predict(test_x[:3])
print(pred)
print('실제 정답: ', test_y[:3])
