from sklearn.datasets import load_digits
import matplotlib as plt
import matplotlib.pyplot as pltt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
mnist = load_digits() # 0-9 손글씨 숫자 데이터 셋

print(mnist['data'][:3])
print(len(mnist['data']))
print(mnist['target'])
print(len(mnist['target']))

features = mnist['data']
labels = mnist['target']

from sklearn.model_selection import cross_validate # 교차검증
RFmodel = RandomForestClassifier()
RF_scores = cross_validate(RFmodel,features, labels, cv=10) # 10-fold 교차검증
print(RF_scores['test_score']) # 랜덤 포레스트 앙상블 검증평가 점수

DT_scores = cross_validate(tree.DecisionTreeClassifier(), features, labels, cv=10) # 의사결정 트리 10-fold 교차검증
print(DT_scores['test_score']) # 의사결정트리 검증평가 점수

import numpy as np
print('random_forest accuracy: ', np.mean(RF_scores['test_score']))
print('decision tree accuracy: ', np.mean(DT_scores['test_score']))
# 랜덤 포렛트 앙상블이 별도의 하이퍼 파라미터 설정없는 의사결정 트리보다 월등히 높은 성능을 발휘함

import pandas as pd
df = pd.DataFrame({'random_forest':RF_scores['test_score'],
                   'decision_tree': DT_scores['test_score']})

print(df)
df.plot()
pltt.savefig('아브래쨔')