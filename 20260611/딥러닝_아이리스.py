import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
iris = load_iris()
# print(iris)
#print(iris['feature_names'])

Iris_Data = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                         columns=['sepal_len','sepal_wd',
                                  'petal_len','petal_wd','target'])
print(iris['target'])

# iris['data'] == 4개의 특성테이터 를 train_x
# iris['target'] ==> train_y

# 'setosa','versicolor', 'virginica' 3가지 붓꽃 클래스 분류 (다중분류)

# 1. 데이터 전처리
print(iris['feature_names'])

Iris_Data = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                         columns=['sepal_len','sepal_wd',
                                  'petal_len','petal_wd','target'])
print(Iris_Data)
print(iris['target'])
iris_y = Iris_Data[['target']]
iris_x = Iris_Data[['sepal_len','sepal_wd',
                                  'petal_len','petal_wd']]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(iris_y)
print(y_encoded)

# categorical_crossentropy() ==> 정답이 원-한 인코딩 상태여야함
from tensorflow.keras.utils import to_categorical
y_onehot = to_categorical(y_encoded)
print(y_onehot)


# # 2. 데이터 분할
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = \
    train_test_split(iris_x, y_onehot, random_state=42)
print(test_x)
print(train_x.shape)
print(test_x.shape)

# 3. 스케일 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)
import joblib
joblib.dump(scaler, 'iris_scaler.plk')

# 4. 다중분류 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
multi_model = Sequential()
multi_model.add( Dense(units=512, input_dim=4, activation='leaky_relu'))
multi_model.add( Dense(units=256, activation='leaky_relu'))
multi_model.add( Dense(units=128, activation='leaky_relu'))
multi_model.add( Dense(units=32, activation='leaky_relu'))
multi_model.add( Dense(units=8, activation='leaky_relu'))
multi_model.add( Dense(units=3, activation='softmax'))

multi_model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['accuracy'])

# 5. 학습

multi_model.fit(train_scaled, train_y, batch_size = 1, epochs=50000, verbose=1)
print('Test acc: ', multi_model.evaluate(test_scaled, test_y)[1])

multi_model.save('iris_multi_clf.keras')
# 별도) 스케일과 모델을 별도 저장
joblib.dump(scaler, 'iris_scaler.pkl')
multi_model.save('iris_multi_clf.keras')
joblib.dump(le, 'iris_label_encoder.pkl')
# 단) categorical_crossentrpy만 사용