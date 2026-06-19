import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# 오토인코더 = 훈련데이터 = 학습데이터
(train_x, _ ), (train_y, _ ) = fashion_mnist.load_data()
print(len(train_x))
print(train_x.shape)