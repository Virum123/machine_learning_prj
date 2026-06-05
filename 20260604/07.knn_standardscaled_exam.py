import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split  # 데이터셋 분할 기능

# 1. 학습시킬 데이터셋 준비
# 도미데이터 ( 캐글Fish Market 데이터 참조 )
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]

bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어데이터 ( 캐글Fish Market 데이터 참조 )
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = np.column_stack( (length, weight) )
print(fish_data[:5])
fish_target = np.concatenate( (np.ones(35), np.zeros(14)) )
print(fish_target)

# 원 데이터를 train / test로 분할 ( 기본 shuffle )
train_x, test_x, train_y, test_y = train_test_split(fish_data,fish_target, stratify=fish_target, random_state=42)

print(train_x)
mean = np.mean(train_x, axis=0) # 평균계산
print(mean)
std = np.std(train_x, axis=0) # 표준편차 계산
print(std)

# 표준점수 정규화 공식 ==> ( 각 특성데이터 - 평균) / 표준편차
train_scaled = (train_x - mean) / std
print( train_scaled ) 
newdata = ([25, 150] - mean) / std
print(newdata)

# 스케일 정규화 된 특성 데이터(입력 데이터) 를 이용해서 모델 학습
# 모델 준비
knn = KNeighborsClassifier()
# 모델 학습
knn.fit(train_scaled, train_y) # y는  정규화해도 0이랑 1이잖어
# 모델 성능 평가
test_scaled = (test_x - mean) / std
# 모델 예측 > predict()
newdata = ([[25,150]]/-mean)/std
print(newdata)
# new1 = newdata.reshap(1,2)또는 [newdata] 로 차원을 변경할 수 있음
pred = knn.predict(newdata)
print(newdata.shape)
print(test_scaled)
print('acc :', knn.score(test_scaled, test_y))
# plt.scatter(newdata[0],newdata[1], marker='D', c='red')
# plt.scatter(train_scaled[:,0], train_scaled[:,1])
# plt.savefig('knn_newfish.jpeg')