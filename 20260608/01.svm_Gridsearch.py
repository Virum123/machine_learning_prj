from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


iris = load_iris()

iris_df = pd.DataFrame(np.column_stack( [iris['data'], iris['target']]),
                       columns=['sepal_len', 'sepal_wd', 'petal_len', 'petal_wd'])
x_petal = iris[['petal_len', 'petal_ed']]
print(x_petal.sample(5))
y_target = iris_df[['target']]

from sklearn.model_selection import GridSearchCV
from sklearn import svm

def svc_param_selection(x, y, nfolds):
    svm_parameters = [
        {'kernel':['rbf']},
        {'gamma':[0.1,0.3,0.5,0.7,1.0]},
        {'C':[0.3,0.7,1.1,1.3,1.5]}
    ]
    # 10번 교차 검증
    clf = GridSearchCV(svm.SVC(), svm_parameters, cv=nfolds)
    clf.fit(x,y) # 10번 교차검증 진행 후 마지막 최적 하이퍼 파라미터로 업데이트

    