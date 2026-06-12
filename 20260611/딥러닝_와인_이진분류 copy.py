

# 와인 데이터 이진 딥러닝 분류 모델 설계
# 학습 까지
import numpy as np
import pandas as pd
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns',500)

wine_df = pd.read_csv('/home/kk30100/deeplearning_prj/20260611/wine_dataset.csv')
print(wine_df)
# Survived 컬럼데이터를 타깃으로 활용( 0, 1)
# 머신러닝 sklearn 은 타깃이 문자열 이어도 성능평가 가능
# Survived 컬럼열 데이터를 변경
# 0 ==> fail,  1 ==> suvival
print(wine_df.info())
# wine_df['Survived'] = wine_df['Survived'].map({1:'suvival',0:'fail'})
print(wine_df.head())

# 모델 입력 데이터 준비
# gender, Age, Pclass 3가지 컬럼 데이터가 생존/비생존에 많은 영향을 미침
print(wine_df['style'])
# 'red'를 0 으로  , 'white'를 1  로 변경
wine_df['style'] = wine_df['style'].map({'red':0, 'white':1})
print(wine_df['style'].value_counts())


# # Age, gender, Class_1, Class_2  이 4개 컬럼 데이터를 모델 입력 데이터로 사용
# # 'Survived' 컬럼은 모델 정답(target) 데이터로 사용
# wine_df_x = wine_df[['gender','Age','Class_1','Class_2']]
# print(wine_df_x)

feature_cols = [col for col in wine_df.columns if col != 'style']

wine_x = wine_df[feature_cols]
wine_y = wine_df['style']

print("입력데이터 체크 : ")
print(wine_x)
print("타깃데이터 체크 : ")
print(wine_y)

# wine_df.info() # 각 특성데이터 타입 체크

# # train / test 분리 해서 사용
from sklearn.model_selection import train_test_split

train_x, x_val, train_y, y_val = \
    train_test_split(wine_x, wine_y,test_size=0.2, random_state=42)

print(train_x[:10])

# # 특성데이터의 스케일 변환(정규화) ==> 표준점수 정규화 ( 각특성 - 평균 / 표준편차 )
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_x) # train 데이터를 정규화 하는 방법을 학습하고 학습이 끝나면
# 변환 작업을 수행 
# test데이터셋은 transform() 만 해서 적용만 해야 함
test_scaled = scaler.transform(x_val)
print("train scaled : ")
print(train_scaled[:10])
print("train_scaled shape:", train_scaled.shape)
print("test_scaled shape:", test_scaled.shape)

# # 입력 특성데이터 13개
# # batch_size ==> 16
# # epochs ==> 200
# # 딥러닝 모델 설계는 교재 참조

# # 모델 설계 후 fit()까지 진행

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(input_shape=(train_scaled.shape[1],), units=8, activation='relu'))
# 은닉층
model.add(Dense(units=4, activation='relu'))
# 출력층
# 이진 분류라 Dense(1) + sigmoid 사용
model.add(Dense(1, activation='sigmoid'))



# =========================
# 7. 모델 컴파일
# =========================
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

# =========================
# 8. 모델 학습
# =========================
history = model.fit(train_x, train_y, validation_data=(x_val, y_val), epochs=50, batch_size = 50)

# =========================
# 9. 모델 평가
# =========================
score = model.evaluate(test_scaled, y_val)
print('Testacc: ', score[1])

model.summary()

model.save("wine_model.keras")

print(history.history.keys())

print(history.history['loss'])
print(history.history['accuracy'])
print(history.history['val_loss'])
print(history.history['val_accuracy'])

import matplotlib.pyplot as plt
plt.plot(history.history['loss']) # 훈련셋 손실
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.savefig('집에가고싶다') 
# 모델 패쓰를 만들어서 모델 체크포인트 설정하기 CALLBACK에서 MODELCHECKPOINT 로 이름은 wine_best.keras


# 콜백은 무엇이고 조기종료, 드랍아웃, 체크포인트 뭐시기