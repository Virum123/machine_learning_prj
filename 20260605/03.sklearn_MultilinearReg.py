import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
# 데이터 분리
train_x, test_x, train_y, test_y = train_test_split(perch_length, perch_weight, random_state=42)

# 차원 변경
train_x = train_x.reshape(-1,1)
test_x = test_x.reshape(-1,1)
# print(train_x[:5] )
# print(train_x[:5]**2 )


train_poly = np.column_stack( (train_x**2, train_x) )
print(train_poly[:5])
print(train_poly.shape)
test_poly = np.column_stack( (test_x**2, test_x) )
print(test_poly.shape)

multi_lrmodel = LinearRegression()
multi_lrmodel.fit(train_poly, train_y)
print('acc: ', multi_lrmodel.score(test_poly, test_y))
# 가중치, 편향 출력
print(multi_lrmodel.coef_, multi_lrmodel.intercept_)

# 예측
# 길이가 30인 농어의 무게를 예측
pred30 = multi_lrmodel.predict([[900, 30]])
print('pred: ', pred30)
print( multi_lrmodel.coef_[0]*900 + multi_lrmodel.coef_[1]*30 + multi_lrmodel.intercept_ )

plt.scatter(train_x, train_y)

xpoint = np.arange(15, 50) # [15...49]
plt.plot(xpoint, 1.01*xpoint**2 - 21.*xpoint + 116.05)

plt.savefig('농어곡선')