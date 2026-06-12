from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = load_iris()

iris_df = pd.DataFrame(
    np.column_stack([iris['data'], iris['target']]),
    columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wd', 'target']
)

x_petal_len_wd = iris_df[['petal_len', 'petal_wd']]
y_target = iris_df['target']   # 1차원 Series로 두는 게 편함

cost = 0.3
g = 0.7

train_x, test_x, train_y, test_y = train_test_split(
    x_petal_len_wd,
    y_target,
    random_state=42
)

svm_model = svm.SVC(C=cost, kernel='rbf', gamma=g)

svm_model.fit(train_x, train_y)

print('train acc:', svm_model.score(train_x, train_y))
print('test acc:', svm_model.score(test_x, test_y))

pred = svm_model.predict([[4.7, 1.7]])
print('예측값:', pred)
print('예측 품종:', iris['target_names'][int(pred[0])])

lnames = iris['target_names']
markers = ['o', '^', 's']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 6))

# 결정경계용 좌표 만들기
x_min, x_max = x_petal_len_wd['petal_len'].min() - 0.5, x_petal_len_wd['petal_len'].max() + 0.5
y_min, y_max = x_petal_len_wd['petal_wd'].min() - 0.5, x_petal_len_wd['petal_wd'].max() + 0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = svm_model.predict(grid_points)
Z = Z.reshape(xx.shape)

# 결정경계 배경
plt.contourf(xx, yy, Z, alpha=0.2)

# train 데이터 산점도
for i in sorted(train_y.unique()):
    idx = train_y == i
    
    plt.scatter(
        train_x.loc[idx, 'petal_len'],
        train_x.loc[idx, 'petal_wd'],
        c=colors[int(i)],
        marker=markers[int(i)],
        label=lnames[int(i)] + ' (train)',
        s=80,
        alpha=0.6
    )

# test 데이터도 같이 표시
for i in sorted(test_y.unique()):
    idx = test_y == i
    
    plt.scatter(
        test_x.loc[idx, 'petal_len'],
        test_x.loc[idx, 'petal_wd'],
        c=colors[int(i)],
        marker=markers[int(i)],
        edgecolors='black',
        label=lnames[int(i)] + ' (test)',
        s=100,
        alpha=1
    )

# 직접 예측한 점 표시
plt.scatter(
    4.7, 1.7,
    c='yellow',
    edgecolors='black',
    marker='*',
    s=300,
    label='prediction point'
)

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('SVM Classification with Iris Petal Features')
plt.legend()
plt.grid(True)

plt.savefig('sf.png')
plt.show()