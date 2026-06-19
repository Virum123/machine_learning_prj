from sklearn import datasets
from sklearn import tree # 의사결정 트리모델
# 불순도가 낮아지는 방향으로트리를 성장시켜서 분류
from sklearn.neighbors import KNeighborsClassifier # knn 분류모델
# K 개의 최근접 이웃이 뭐냐?
from sklearn.svm import SVC # SVM
# ==> 결정 경계를 활용한 분류모델
# 나이브베이지 ==> 조건부 확률
from sklearn.ensemble import VotingClassifier # 보팅 분류
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mnist = datasets.load_digits() # 0~9 손글씨셋
print(mnist)

features = mnist['data']
labels = mnist['target']

print(len(features))
print(len(labels))

train_x, test_x, train_y, test_y = \
    train_test_split(features, labels, test_size=0.2)

# tree 모델
model_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=8,
                                         max_features=32, random_state=46)
model_tree.fit(train_x, train_y)
print('acc : ', model_tree.score(test_x, test_y)) # acc : 0.8166666666666667

# KNN
model_knn = KNeighborsClassifier( n_neighbors=299 )
model_knn.fit(train_x, train_y)
print('knn acc : ', model_knn.score(test_x, test_y)) #0.085

# svm모델
model_svc = SVC (C=0.1, gamma=0.003, probability=True, random_state=46)
model_svc.fit(train_x, train_y)
print('SVC acc : ', model_svc.score(test_x, test_y))

# 하드보팅 정확도
hardvoting_model = VotingClassifier(estimators=[
    ('decison_tree', model_tree ),
    ('knn',model_knn),
    ('svm', model_svc)
], weights=[1,1,1], voting='hard'
)
hardvoting_model.fit(train_x, train_y) # 하드보팅 모델 학습
print('hardvoting acc : ', hardvoting_model.score(test_x, test_y))

softvoting_model = VotingClassifier(estimators=[
    ('decison_tree', model_tree ),
    ('knn',model_knn),
    ('svm', model_svc)
], weights=[1,1,1], voting='soft'
)

softvoting_model.fit(train_x, train_y) # 하드보팅 모델 학습
print('softvoting acc : ', softvoting_model.score(test_x, test_y))