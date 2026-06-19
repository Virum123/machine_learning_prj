import numpy as np
from tensorflow.keras.datasets import fashion_mnist
# failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED : GPU 메모리 할당오류
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)
##################################
# 오토인코더 경우 훈련데이터를 타깃으로 이미지 재건 출력 학습모델(비지도) 로 타깃 데이터 불필요
(X_train, _), (X_test , _) = fashion_mnist.load_data()
print(len(X_train) ) # 60000
print(X_train.shape) # (60000, 28, 28)
# X_train = X_train.reshape(60000, 28, 28, 1) # CNN 입력을 위한 차원 변경
# X_test = X_test.reshape(10000, 28, 28, 1) # CNN 입력을 위한 차원 변경
# print(X_train.shape) # (60000, 28, 28, 1) : (샘플수, 3차원 이미지 )
# print(X_test.shape) # (10000, 28, 28, 1) : (샘플수, 3차원 이미지 )
# # 데이터 정규화
# X_train = X_train / 255.0
# X_test = X_test / 255.0

# from tensorflow.keras.models import load_model 
# autoencoder_model = load_model('best-LSTM-model.keras')

# #X_test 데이터 예측(복원) 
# ae_predict_imge = autoencoder_model.predict(X_test) # (10000, 28, 28, 1)
# print(ae_predict_imge.shape)
# print(ae_predict_imge[0])

# import matplotlib.pyplot as plt 
# num = 5 # 원본과 5개 비교
# plt.figure(figsize=(15,7))
# for i in range(num): # 원본 이미지
#     ax1 = plt.subplot(2, num, i+1) # (2, 5) 플롯 중 윗줄 서브플롯
#     ax1.imshow( X_test[i].reshape(28,28), cmap='gray') 
#     ax1.set_title('original_image %d' %i) # 서브플롯에 타이틀 추가
#     ax1.axis('off')
#     # 복원 이미지
#     ax2 = plt.subplot(2, num, i + num + 1) # (2, 5) 플롯 중 아래줄 서브플롯
#     ax2.imshow(ae_predict_imge[i].reshape(28, 28) , cmap='gray') 
#     ax2.set_title('autoenc_imge %d' % i) # 서브플롯에 타이틀 추가
#     ax2.axis('off')
# plt.savefig('복원옷')