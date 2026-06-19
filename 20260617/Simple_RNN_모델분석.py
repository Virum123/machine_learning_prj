from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x)
print(x.shape)
x = x.reshape(4,3,1) # (3 timesteps, 1 입력 차원수) ==> 4개의 샘플
print(x)

rnnmodel = Sequential()
rnnmodel.add( SimpleRNN(10,return_sequences=False, input_shape=(3,1)) )# 3개 timesteps , 1 입력차원수
rnnmodel.add( Dense(1) ) # default : linear , 별도 활성화함수 없이 입력 뉴런과 가중치 계산결과가 그대로 출력
rnnmodel.summary()

rnnmodel.compile(optimizer='adam', loss='mse', metrics= ['mse'])
rnnmodel.fit(x,y, epochs=1000, batch_size=1) # 데이터가 작아서 1000으로 한거임. 
# 모델 학습
print(rnnmodel.predict(x)) 

pre_input = np.array([6,7,8])
pre_input = pre_input.reshape((1,3,1))

pre_out = rnnmodel.predict(pre_input)
print(pre_out)