# import numpy as np
# import joblib

# from tensorflow.keras.models import load_model

# # =========================
# # 1. 저장된 객체 불러오기
# # =========================

# # 학습 때 저장한 스케일러 불러오기
# scaler = joblib.load('/home/kk30100/deeplearning_prj/fish_scaler.pkl')

# # 학습 때 저장한 라벨 인코더 불러오기
# le = joblib.load('/home/kk30100/deeplearning_prj/fish_label_encoder.pkl')

# # 학습된 딥러닝 모델 불러오기
# multi_model = load_model('/home/kk30100/deeplearning_prj/20260611/fish_multi_clf.keras')


# # =========================
# # 2. 새 물고기 데이터 입력
# # =========================
# # 입력 순서 중요
# # Weight, Length, Diagonal, Height, Width

# new_fish = [[242.0, 25.4, 30.0, 11.52, 4.02]]


# # =========================
# # 3. 스케일링
# # =========================
# # 새 데이터에는 fit_transform() 금지
# # 학습 때 저장한 scaler로 transform만 해야 함

# new_scaled = scaler.transform(new_fish)


# # =========================
# # 4. 예측
# # =========================

# pred_prob = multi_model.predict(new_scaled)

# print("클래스별 확률:")
# print(pred_prob)


# # =========================
# # 5. 가장 확률 높은 클래스 선택
# # =========================

# pred_class = np.argmax(pred_prob, axis=1)

# print("예측 클래스 번호:")
# print(pred_class)


# # =========================
# # 6. 숫자 클래스를 물고기 이름으로 변환
# # =========================

# pred_name = le.inverse_transform(pred_class)

# print("예측 물고기 종류:")
# print(pred_name[0])


# fishclass = ['Bream', 'Parkki', 'Perch', 'Roach', 'Smelt', 'Whitefish']

from tensorflow.keras.models import load_model
import joblib
import numpy as np

np.set_printoptions(precision=8, suppress=True)
fish_bestmodel = load_model('fish_multi_clf.keras')
fish_bestmodel.summary()

newfish_scaler = joblib.load('fish_scaler.pkl')
fishclass = ['Bream', 'Parkki', 'Perch','Pike' 'Roach', 'Smelt', 'Whitefish'] 

new_fish_data = [[50,60,70,80,90],
                 [57,99,256,1,30],
                 [1,23,10,8,900]]
new_fish_data_scaled = newfish_scaler.transform(new_fish_data)
print(new_fish_data_scaled)
pred = fish_bestmodel.predict(new_fish_data_scaled)
print(pred)

print( np.argmax(pred, axis = 1))

fishclass = np.array(['Bream', 'Parkki', 'Perch','Pike', 'Roach', 'Smelt', 'Whitefish'] )
print(fishclass[np.argmax(pred, axis=1)])