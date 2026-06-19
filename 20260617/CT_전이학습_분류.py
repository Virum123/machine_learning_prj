train_dir = '/home/kk30100/deeplearning_prj/20260617/Covid19_CT_Image_dataset/train'
test_dir = '/home/kk30100/deeplearning_prj/20260617/Covid19_CT_Image_dataset/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batch_size = 20
image_size = 224
# train 이미지 증강 유형 생성
train_image_generator = ImageDataGenerator( rescale = 1.0/255.,
                   rotation_range = 180,
                   width_shift_range=0.2,
                   height_shift_range=0.2,
                   horizontal_flip=True,
                   vertical_flip=True )

# train image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
train_data_gen = train_image_generator.flow_from_directory(
    train_dir, # 불러올 이미지 경로
    batch_size = batch_size,
    shuffle = True,
    # 디렉토리 내부 이미지 불러올떄 어떤 형식으로 라벨링해서 불러올건지
    class_mode = 'categorical',
    target_size = (image_size, image_size)
)

# test 이미지 증강 유형 생성
test_image_generator = ImageDataGenerator( )

# train image를 읽어 들이면서 image를 증강 시켜주는 제너레이터 생성
test_data_gen = test_image_generator.flow_from_directory(
    test_dir, # 불러올 이미지 경로
    batch_size = batch_size,
    shuffle = False,
    # 디렉토리 내부 이미지 불러올떄 어떤 형식으로 라벨링해서 불러올건지
    class_mode = 'categorical',
    target_size = (image_size, image_size)
)

# 디렉토리 별 자동 라벨링 정보를 갖고 있음
print(train_data_gen.class_indices) # {'Covid': 0, 'Normal': 1}

class_levels = train_data_gen.class_indices.keys()
print(class_levels)

# vgg16 모델의 가중치를 가져와서 전이학습 하는 모델 설계
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense

# VGG16 모델의 Top층(FC layer) 의 가중치는 가져오지마. 재설계 할거야
vgg16_layer = vgg16.VGG16(weights='imagenet', include_top=False,
            input_shape=(image_size, image_size, 3))
vgg16_layer.summary()

for layer in vgg16_layer.layers:
    layer.trainable = False # Vgg16 모델의 하단(특징 추추역할 레이어)는 학습되지 마세요

newmodel = Sequential()
newmodel.add(vgg16_layer)

# 상단층 재설계
newmodel.add( Flatten() )
newmodel.add( Dense(units=1024, activation='leaky_relu') )
newmodel.add( Dropout(0.3) )
newmodel.add( Dense(units=2, activation='softmax') )

newmodel.summary()
import tensorflow as tf
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) # 0.00001
newmodel.compile(loss='categorical_crossentropy',  optimizer = optimizer,
                 metrics = ['accuracy'])

import numpy as np
print ( int(np.ceil(train_data_gen.samples/train_data_gen.batch_size)))
# print(train_data_gen.samples)
# print(test_data_gen.batch_size)

newmodel.fit(test_data_gen, 
             steps_per_epoch =  int(np.ceil(train_data_gen.samples/train_data_gen.batch_size)),
             epochs = 30,
             validation_data = test_data_gen,
             validation_steps = int(np.ceil(test_data_gen.samples/test_data_gen.batch_size)),
             verbose = 1
            )

newmodel.save('vgg16_newmodel.h5')

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg16

newmodel = load_model('vgg16_newmodel.h5')
newmodel.summary()

from tensorflow.keras.preprocessing import image
import numpy as np
pred_list = [] # 예측 결과 저장 리스트
# test 이미지 데이터 로딩 및 예측 수행

def predict_vgg16_newmodel(newmodel, filename): # 파일별 예측 결과 저장 함수
    img = image.load_img(filename, target_size=(224,224)) # 파일 이미지 로드
    img_arr = image.img_to_Array(img) # image 데이터 넘파이 배열로 변환
    image_reshape = img_arr.reshape((1, 224, 224, 3))
    image_input = vgg16.preprocess_input(image_reshape) # vgg16 모델입력 전처리

    # {'Covid':0, 'Normal':1}
    pred = newmodel.predict(image_input, batch_size=1) # 해당 이미지파일 예측
    #print(pred)
    class_list = ['covid19', 'normal']
    print('pred result:', class_list[np.argmax(pred)]) # 예측 최대 추정치 인덱스 추출
    pred_list.append( class_list[np.argmax(pred)]) # 예측 결과 list에 저장

filenamelist = os.listdir(test_dir) # 디렉토리 내부의 모든 파일 정보 리스트 반환
print(filenamelist)

file_totalinfo = []
for file in filenamelist:
    file_totalinfo.append(test_dir+file)

print(file_totalinfo)

for imagefile in file_totalinfo:
    predict_vgg16_newmodel(newmodel, imagefile) # 예측 함수 호출

class_name='covid19' # 코로나 이미지로 예측 수행해서 분류이름을 covid19로 사용

import pandas as pd
from sklearn.metrics import accuracy_score # 예측 성능 평가

df = \
pd.DataFrame({'True_data':[class_name]*len(file_totalinfo), 'Pred_Data': pred_list, 'filename': filenamelist})

print(df)

print('accuracy: #.3f'%accuracy_score(df['True_Data'], df['Pred_Data']))