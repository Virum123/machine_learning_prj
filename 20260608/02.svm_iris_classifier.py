from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm # svm 모델 추가

iris = load_iris()

# print(iris)

iris_df = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                       columns=['sepal_len','sepal_wid', 'petal_len', 'petal_wd', 'target']) #iris data 랑 target list 를 묶어서 df로 

print(iris_df)


x_petal_len_wd= iris_df[['petal_len', 'petal_wd']] #train test 할때의 x set
print(x_petal_len_wd.sample(5))

y_target = iris_df[['target']] # train test 의 y set
print(y_target.sample(5))

cost = 0.3
g = 0.7

# train/test split
train_x, test_x, train_y, test_y = train_test_split(x_petal_len_wd, y_target, random_state=42)

print(test_x[:5])
print(test_x[:5])

svm_model = svm.SVC(C = cost, kernel='rbf', gamma=g)

# train 데이터 활용해서 모델 학습
svm_model.fit(train_x, train_y.values.ravel())

# print('train acc :', svm_model.score(train_x.values.ravel(), train_y.values.values.ravel()))
# print('test acc :', svm_model.score(test_x.values.ravel(), test_y.values.ravel()))

pred = svm_model.predict([[4.7, 1.7]])
print(pred)

lnames = iris['target_names']
markers = ['o', '^', 's']
colors = ['blue', 'green', 'red']

for i in set(train_y['target']):
    idx = np.where(train_y['target'] == i)
    print(idx[0])
    plt.scatter(train_x.iloc[idx]['petal_len'],train_x.iloc[idx]['petal_wd'],
               c=colors[int[i]], marker = markers[int[i]], label = lnames[int(i)] + '(train)', s=80, alpha=0.3 )


plt.savefig('sf')