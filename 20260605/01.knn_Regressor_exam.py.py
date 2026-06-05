import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

#농어길이데이터 ( 캐글Fish Market 데이터참조)
perch_length=np.array([8.4,13.7,15.0,16.2,17.4,18.0,18.7,19.0,19.6,20.0,21.0,
21.0,21.0,21.3,22.0,22.0,22.0,22.0,22.0,22.5,22.5,22.7,
23.0,23.5,24.0,24.0,24.6,25.0,25.6,26.5,27.3,27.5,27.5,
27.5,28.0,28.7,30.0,32.8,34.5,35.0,36.5,36.0,37.0,37.0,
39.0,39.0,39.0,40.0,40.0,40.0,40.0,42.0,43.0,43.0,43.5,
44.0])

# 농어무게데이터 (캐글FishMarket 데이터참조)
perch_weight=np.array([5.9,32.0,40.0,51.5,70.0,100.0,78.0,80.0,85.0,85.0,110.0,
115.0,125.0,130.0,120.0,120.0,130.0,135.0,110.0,130.0,
150.0,145.0,150.0,170.0,225.0,145.0,188.0,180.0,197.0,
218.0,300.0,260.0,265.0,250.0,250.0,300.0,320.0,514.0,
556.0,840.0,685.0,700.0,700.0,690.0,900.0,650.0,820.0,
850.0,900.0,1015.0,820.0,1100.0,1000.0,1100.0,1000.0,
1000.0])

# 길이에 따른 무게를 예측할 것
# 정답(타깃, y) ==> 농어의 무게
# perch_data = np.stack(perch_length, perch_weight)

kn = KNeighborsRegressor(3) # 최적의 K값을 찾는게 어려움

train_x, test_x, train_y, test_y = train_test_split(perch_length, perch_weight, random_state=42)
# train, test 가 하나의 짝임을 알아둘것
print(len(train_x))
print(len(test_x))
print(train_x.shape)
print(test_x.shape)

# 1차원 shape을 2차원 shape으로 변경해서 사용
train_x = train_x.reshape(-1, 1)
test_x = test_x.reshape(-1, 1) # -1 맞춤 설정
print(train_x.shape)
print(test_x.shape)
# 학습
kn.fit(train_x, train_y)
# 훈련 /> k=3으로 변경시
print(kn.score(test_x, test_y)) # 테스트 성능 0.992 > 0.97
print(kn.score(train_x, train_y)) # 훈련 성능 0.996 > 0.98 이게 옳게된 모델
# 예측
test_pred = kn.predict(test_x)
print(test_pred)

# from sklearn.metrics import mean_absolute_error

# mae = mean_absolute_error(test_y, test_pred) # |정답 - 예측치| 평균
# print('mae: ', mae)

# 농어의 길이가 40인 농어의 무게를 예측하시오
pre_40 = kn.predict([[40]])
print(pre_40)
pre_x = [[80],[120]]
pre_two = kn.predict(pre_x)
print(pre_two) # [1033.33333333 1033.33333333] 똑같이 나옴, KNN모델의 문제

plt.scatter(train_x, train_y) # 80과 120은 대응값이나 유사값이 동일하니까 최대값으로 출력되는구나
pre_50 = kn.predict([[50]])
plt.scatter(50, pre_50, marker= '^', c='red')
plt.savefig('농어')