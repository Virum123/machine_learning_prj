import pandas as pd
from sklearn.datasets import load_iris # 데이터셋
from sklearn.model_selection import train_test_split # 데이터셋 분리

from sklearn.naive_bayes import GaussianNB

#  나이브 베이즈 ==>
from sklearn.naive_bayes import GaussianNB # 데이터 특징 가우시안 정규분포를 취할 떄
# 분류 모델로 사용 ==> GaussianNB
from sklearn import metrics # 혼동행렬
from sklearn.metrics import accuracy_score

dataset = load_iris()
print(dataset)

iris_df = pd.DataFrame(dataset['data'], columns=dataset['feature_names'])
print(iris_df)
iris_df['target'] = dataset['target']
print(iris_df)

# 타깃 시각화를 위해서
# 0 =='setosa', 1 ==> 'versicolor', 2 ==> 'virsinica'

iris_df['target'] = iris_df['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

# setosa DF
setosa_DF = iris_df.loc[iris_df['target'] == 'setosa'].copy()
# 데이터 프레임 접근할떈 loc, iloc로
# versicolor DF
versicolor_DF = iris_df.loc[iris_df['target'] == 'versicolor'].copy()
# virginica DF
virginica_DF = iris_df.loc[iris_df['target'] == 'virginica'].copy()

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1,3, figsize = (15,7))

sns.histplot(data=setosa_DF, x='sepal length (cm)', kde = True, ax=axes[0])

plt.savefig('iris_histplot.jpeg')