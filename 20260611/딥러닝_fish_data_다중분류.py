import numpy as np
import pandas as pd

# e지수 표현하는 과학적표기 대신 소수점 이하 8자리까지 표현
np.set_printoptions(precision=8, suppress=True)
np.set_printoptions(threshold=np.inf) # 무한으로 출력합니다.

fishdf = pd.read_csv('/home/kk30100/deeplearning_prj/20260611/fish_data.csv')
print(fishdf)
print(fishdf['Species'].unique()) # 물고기 종이 몇 종?
# 7개 물고기 중 어떤 물고기야?? (다중분류)
fishdf.info()


# Species 컬럼은 정답 데이터이므로 따로 분리
fish_target = fishdf['Species']

# Species를 제외한 나머지 컬럼은 입력 데이터로 사용
fish_train = fishdf[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

# 라벨( 문자열)을 수치형태로 반환
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(fish_target)
print(y_encoded)

# categorical_crossentropy() ==> 정답이 원-한 인코딩 상태여야함
from tensorflow.keras.utils import to_categorical
y_onehot = to_categorical(y_encoded)
print(y_onehot)

# train / test 분리
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = \
    train_test_split(fish_train, y_onehot, random_state=42)

print(train_x.shape)
print(test_x.shape)

# 특성 데이터에 대한 스케일 조정이 필요함

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x)
test_scaled = scaler.transform(test_x)
import joblib
joblib.dump(scaler, 'fish_scaler.plk')

# 반대로 읽어들일 떄는 joblib.load('fish_scaler.plk')

# 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
multi_model = Sequential()
multi_model.add( Dense(units=10, input_dim=5, activation='leaky_relu'))
multi_model.add( Dense(units=7, activation='softmax'))

# multi_model.summary()

multi_model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['accuracy'])

# 모델 학습
multi_model.fit(train_scaled, train_y, batch_size = 1, epochs=500, verbose=1)

# 모델 성능평가
print('Test acc: ', multi_model.evaluate(test_scaled, test_y)[1])

multi_model.save('fish_multi_clf.keras')

import joblib
##### GPT
# 스케일러 저장
joblib.dump(scaler, 'fish_scaler.pkl')

# 라벨 인코더 저장
joblib.dump(le, 'fish_label_encoder.pkl')

# 모델 저장
multi_model.save('fish_multi_clf.keras')