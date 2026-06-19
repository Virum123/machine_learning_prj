import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input

(train_x, _ ),(test_x, _ ) = fashion_mnist.load_data() # (28,28) 패션 관련 흑백 이미지

print(len(train_x))

train_x = train_x.reshape(-1, 28, 28, 1)
print(train_x.shape)

test_x = test_x.reshape(-1, 28, 28, 1)
print(test_x.shape)
print(train_x[0])

train_x = train_x/255.0
test_x = test_x/255
# 모델 설계
autoencoder_model = Sequential()
autoencoder_model.add(Input( (28,28,1)))

# 인코더
autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same',
                            activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2)))
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same',
                            activation='leaky_relu'))
autoencoder_model.add(MaxPooling2D(pool_size=(2,2))) 
autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), strides=2, padding='same', activation='leaky_relu'))
autoencoder_model.summary()

# 디코더

autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3 ), padding='same',activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=8, kernel_size=(3,3), padding='same', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())


autoencoder_model.add(Conv2D(filters=16, kernel_size=(3,3), padding='valid', activation='leaky_relu'))
autoencoder_model.add(UpSampling2D())

autoencoder_model.add(Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid'))
autoencoder_model.summary()

autoencoder_model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                          metrics=['accuracy'])


autoencoder_model.save('autoencodermodel.keras') # fit() 시 모든 epochs 가 다 동작
# 조기종료 콜백은 이후에 진행
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 모델 저장
checkpoint_cb = ModelCheckpoint('best-LSTM-model.keras') # h5 대신 keras 사용
# 조기종료
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights = True)

history = autoencoder_model.fit(train_x, train_x, epochs=50, batch_size=64,
                    validation_data=(test_x,test_x),
                    callbacks=[checkpoint_cb, early_stopping_cb])