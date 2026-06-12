from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

iris = load_iris()

# print(iris)

iris_df = pd.DataFrame(np.column_stack([iris['data'], iris['target']]),
                       columns=['sepal_len','sepal_wid', 'petal_len', 'petal_wd', 'target']) #iris data 랑 target list 를 묶어서 df로 

print(iris_df)


x_petal_len_wd= iris_df[['petal_len', 'petal_wd']] #train test 할때의 x set
print(x_petal_len_wd.sample(5))

y_target = iris_df[['target']] # train test 의 y set
print(y_target.sample(5))

# 바로 train 하면 최적의 hyper parameter 가 아닐수 있음으로 최적의 하이퍼 파라미터 갖는 모델을 교차검증을 활용해서 찾음
#==> Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# svc_param_selection(x_petal_len_wd, y_target.values.)
# print(y_target.values.ravel()) # .ravel() 가 왜 들어가냐면, data frame 을 써서 GridsearchCV 를 하면 에러가 날 수 있다.
# 그래서 2차원을 1차원으로 변경하기 위해 .ravel() 이 된다

def svc_param_selection(x,y, nfolds):
    svm_parameters = [
        {'kernel':['rbf'],
        'gamma': [0.1, 0.3, 0.5, 0.7, 1.0],
        'C': [0.3, 0.7, 1, 1.3, 1.5]
         }
    ]
    # 10 번 교차 검증
    clf = GridSearchCV(svm.SVC(), svm_parameters, cv=nfolds) # svm안에 들어가 있는데 SVC 모델
    clf.fit(x,y) # 10번 교차검증 진행하고 마지막 최적하이퍼 파라미터로 업데이트
    print(clf.best_params_) # 최적 파라미터 
    print(clf.best_score_) # 최적 성능
    print(clf.best_estimator_) # 최적 모델

    return clf.best_estimator_


svc_mode = svc_param_selection(x_petal_len_wd, y_target.values.ravel(), 10)

