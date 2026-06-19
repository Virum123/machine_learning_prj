import numpy as np
from tensorflow.keras.layers import Embedding

input_data = np.array([
[3,4,7], [9,2,3], [4,9,499]
])

# input_dim: 변환할 입력갑싀 최대값 ==> maximum integer index + 1
# output_dim: 결과값을 몇 개의 벡터로 지정할지 지정
# input_length: Length of input sequences

embedding = Embedding(input_dim = 500, output_dim=2, input_length=3)
oupout = embedding(input_data)
print(oupout)