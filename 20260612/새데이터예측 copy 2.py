import cv2 # opencv-python
import numpy as np

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)

img = cv2.imread('/home/kk30100/deeplearning_prj/20260612/sandal_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/kk30100/deeplearning_prj/20260612/다운로드.jpg', cv2.IMREAD_GRAYSCALE)

print(img)
print(img2)
print(img.shape)
print(img2.shape)

img_resize = cv2.resize(img, dsize=(28,28), interpolation=cv2.INTER_AREA)
img_resize2 = cv2.resize(img2, dsize=(28,28), interpolation=cv2.INTER_AREA)
print(img_resize)
print(img_resize2)
# cv2.imshow('sandal_resize')
# cv2.imshow('agu_resize')

import matplotlib.pyplot as plt
plt.imshow(img_resize2 , cmap='gray')


img_reverted = cv2.bitwise_not(img_resize2)
print(img_reverted)

plt.imshow(img_resize2 , cmap='gray')
plt.savefig('sandle_reverted.jpeg')

img_reverted = img_reverted / 255.0
print(img_reverted.shape)
img_reverted = img_reverted.reshape(1,28,28,1)
print(img_reverted.shape)

from tensorflow.keras.models import load_model
fsmodel = load_model('/home/kk30100/deeplearning_prj/20260612/mlis_best.keras')
fsmodel.summary()

pred = fsmodel.predict(img_reverted)

classes = ['티', '바지', '스웨터', '드레스','코트', '샌달', '셔츠', '스티커즈', '가방', '앵클부트']
classlf = np.array(classes)
print(classlf[np.argmax(pred, axis=1)])
