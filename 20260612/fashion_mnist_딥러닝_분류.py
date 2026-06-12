import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(train_x, train_y) , (test_x, test_y) = fashion_mnist.load_data()
print(len(train_x), len(test_x)) # 60000, 10000

# 정답의 클래스 분류와 어떤 클래스가 몇 개 있는지 보여주는 것(return_counts)
print(np.unique(train_y , return_counts=True))
print(np.unique(test_y , return_counts=True))


# import matplotlib.pyplot as plt

# print(train_x[0].shape) # (28,28)

# plt.imshow(train_x[0], cmap='gray')
# plt.savefig("슬라임의 장화")

train_scaled = train_x.reshape(-1, 28, 28, 1) / 255.0 # 이미지 데이터 정규화
print(train_scaled.shape)

from sklearn.model_selection import train_test_split

# train / validata 로 분할
train_x, val_x, train_y, val_y =\
    train_test_split(train_scaled, train_y, test_size=0.2, random_state=42)

print(len(train_x))
print(len(val_x))

# 모델 생성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

model = Sequential()
# 첫번째: 필터의 개수
# kernel_size ==> 3 ==> (3 x 3)
# padding = 'same' ==> same padding 이라 하며 다른말로 zero padding 이라고 함
#  첫번재층은 항상 입력 데이터를 생각해서 코딩
# input_shape = (28,28,1) ==> 입력 이미지의 Shape
# Conv. layer 층 추가
model.add( Conv2D(32, kernel_size=3, activation='relu', padding='same', # padding에 따라서 무엇이 달라지는가 same, valid
                  input_shape = (28,28,1)) ) # Conv layer 만들어서 추가할 것

# 풀링층 추가
model.add( MaxPooling2D(2)) # 2 ==> 2x2 필터가 2스트라이드 이동하면서 최대값 선택

# Conv. Layer 층 추가
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))

# 풀링층 추가
model.add( MaxPooling2D(2)) # 2 ==> 2x2 필터가 2스트라이드 이동하면서 최대값 선택


model.add( Flatten())
# FC layer
model.add( Dense(100, activation='relu'))
# 과대 적합 방지
model.add( Dropout(0.4))
model.add( Dense(40, activation='relu'))
# 출력층 설계 ==> 분류하고자하는 클래스의 수 만큼 뉴런이 필요하다
# fashion_mnist 데이터의 라벨(정답)이 ==> 10개 클래스 분류
# 출력층 ==> 활성화 함수 ==> 다중분류 일 경우 ==> softmax
model.add(Dense(10, activation='softmax'))

model.summary()



# 모델 컴파일( optimizer, loss , matrics)

model.compile(loss= "categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])

from tensorflow.keras.utils import to_categorical
y_onehot = to_categorical(train_y)

# 모델 학습
# 콜백 기능 추가해서 best 모델 저장, 조기종료
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath = '/home/kk30100/deeplearning_prj/20260612/mlis_best.keras', monitor='val_loss', verbose=1,
                               save_best_only=True)
# 이름만 쓰면 같은 위치에 저장이 됨, 상대위치를 적어서 원하는 위치에 저장할 수 있음
earlystop_cb = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_x, y_onehot
                     , batch_size=64, epochs=400, verbose =1, callbacks=[checkpointer, earlystop_cb])
