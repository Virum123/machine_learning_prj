from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()
print(iris)
print(iris['data'][:5])
print(iris['feature_names'])
print(type(iris))
print(iris.keys())
# print(iris['target'][:5])
iris_data = np.column_stack( (iris['data'], iris['target']) ) # 데이터를 쌓음 배열연결
print(iris_data[:5])
# 문제
# 위 데이터를 바탕으로 DataFrame을 설계(구현)
# 컬럼명 ==> sepal_len, sepal_wid, petal_len, petal_wid, target

iris_df = pd.DataFrame(
    data=iris_data,
    columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'target']
)
# print(iris_df.head())

# plt.scatter(iris_df['sepal_len'], iris_df['sepal_wid'], c=iris_df['target'])
# plt.savefig('iris.jpeg')

# iris 붓꽃 KNN 분류 모델에서 모델 입력 특성 데이터(train_x)는 
# petal_len, petal_wid 두 가지 특성 데이터만 활용
# 모델 타깃 데이터(train_y)는 target 컬럼의 데이터 활용
iris_train_x = iris_df[['petal_len', 'petal_wid', 'target']].copy() # .values
# print(iris_train_x)

# for i in range(3):
#     plt.scatter( iris_train_x.loc[iris_train_x['target'] == i, :]['petal_len'],
#                 iris_train_x.loc[iris_train_x['target'] ==i, :]['petal_wid'])
    
# plt.savefig('iris.jpeg')
# KNN 모델 준비 ( k = 5 디폴트 사용)
knn = KNeighborsClassifier()
# knn 학습
knn.fit(iris_train_x[['petal_len', 'petal_wid']].values,iris_train_x[ 'target'].values ) 
# sklearn 모델은 pandas DataFrame도 받을 수 있다.
# 하지만 필요하면 .values 또는 .to_numpy()로 NumPy 배열만 꺼내서 넣을 수 있다.
# 단, y값은 2차원이 아니라 1차원으로 넣는 것이 좋다.
print( knn.score(iris_train_x[['petal_len', 'petal_wid']].values, iris_train_x[ 'target'].values) )
# 새로운 데이터 붓꽃 분류( 예측 )
pred = knn.predict([[5.9, 2.3], [3.4, 1.8], [3.1, 4.2]])
print(pred) # [2 1 2] ==> ['virginica', 'versicolor', 'virginica']
# # print(iris.keys())
# print(iris['target_names'])

# word_pred = []
# for i in range(len(pred)):
#     if pred[i] == 0:
#         word_pred.append('setosa')
#     elif pred[i] == 1:
#         word_pred.append('versicolor')
#     elif pred[i] == 2:
#         word_pred.append('virginica')
# print(word_pred)
for x in pred:
    print(iris['target_names'][int(x)])

new_pred = knn.predict([[5.9,2.3]])
print(new_pred)
# petal_len ==> 5.9, petal_sid ==> 2.3인 위치를
# scatter(), '^' 마커로 출력
# 동시에 위 scatter로 출력한 모든 데이터 위에 5.9, 2.3 위치를 출력

for i in range(3):
    plt.scatter( iris_train_x.loc[iris_train_x['target'] == i, :]['petal_len'],
                iris_train_x.loc[iris_train_x['target'] ==i, :]['petal_wid'],)
plt.scatter( 5,1.7, marker = 'P',  c='cyan', s=500, edgecolors='black')

plt.savefig('new_pred.jpeg')