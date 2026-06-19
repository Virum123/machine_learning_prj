from tensorflow.keras.applications import vgg16

vgg16model = vgg16.VGG16()
vgg16model.summary()

# predict_img_dir = '/home/kk30100/deeplearning_prj/20260616/testimage_dataset/'
# import os
# os.chdir('/home/kk30100/deeplearning_prj/20260616/testimage_dataset/') # 경로가 안맞아서 이렇게 해줫다는데 뭔소린지
# print(os.listdir())

# fileinfolist = []
# for file in os.listdir():
#     fileinfolist.append(predict_img_dir + file)

# print(fileinfolist)

# print(fileinfolist[0])

# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array

# img = load_img(fileinfolist[6], target_size = (224, 224))
# # print(img) 어떻게 나오는지 확인
# image = img_to_array(img) # 이미지 객체를 넘파이 배열로 변경
# print(image.shape)

# image = image.reshape(1, 224, 224, 3)
# print(image.shape)

# imaage = vgg16.preprocess_input(image)
# print(image)

# pred = vgg16model.predict(image)
# # print(pred)
# # print(len(pred[0])) 
# # print(vgg16model.classes)

# labels = vgg16.decode_predictions(pred)
# print(labels)