import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/home/kk30100/deeplearning_prj/20260605/basketball_stat.csv')
print(df.head())
print(df.info())
print(df['Pos'].value_counts())

sns.lmplot(x='TRB', y='3P',data=df, fit_reg=False,
           scatter_kws={'s': 150}, markers=['*','P'], hue='Pos')
plt.title('TRB and 3P in 2d plane')
plt.savefig('TRB and 3P in 2d plane')

sns.lmplot(x='BLK', y='3P',data=df, fit_reg=False,
           scatter_kws={'s': 150}, markers=['*','P'], hue='Pos')
plt.title('BLK and 3P in 2d plane')
plt.savefig('BLK and 3P in 2d plane')

df.drop(['2P','AST','STL'],axis=1, inplace=True) # 이름이 []인 열지우기, 0이면 이름이 [] 인 행지우기
print(df.head())

######################################################################################

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.2, random_state=45)
print(train.shape[0])
print(test.shape[0])

# 이제부터 최적의 Knn 파라미터 찾기
# cross_val, score() 활용
# 최적의 k를 찾기 위한 교차검증 수행할 k의범위를
# 3 부터학습데이터절반까지 설정, 홀수로 설정이 적절

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

max_k_range = train.shape[0] // 2 # k 범위 잡아주기
k_list = []
for i in range(3, max_k_range, 2): # 홀수로 잡아줌, 1은 하나라서 넘어가고 3부터 2씩
    k_list.append(i)
print(k_list) # k 값 리스트

x_train= train[['3P','BLK','TRB']]
y_train= train[['Pos']]

cross_validation_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
    cross_validation_scores.append(scores.mean())
# cross_val_score() < 뭐하는애야? 교차검증 점수들을 계산해주는 함수
# cv = 10(훈련 데이터를 10조각낸다, 1-10까지 돌리면서 해당 번호는 검증, 나머진 훈련용)
# 그렇게해서 scores에 점수가 총 10개 들어감
# 그렇기에 아래에서 평균을 내주는거고
# 그럼 k값에 따라서 훈련이 된 점수들을 쭉 뽑아볼 수 있을것

# 여긴 cross_val_score y 값이 1차원이 필요해서 value로 numpy로 변환, ravel로 떙겨옴땡겨옴
print(cross_validation_scores)
plt.figure()
plt.plot(k_list, cross_validation_scores)
plt.xlabel('number of k')
plt.ylabel('Accuracy')
plt.savefig('농구선')

# k = 3 일때 가장 높은 정확도임을 확인함

# 훈련/테스트 데이터 분리

knn= KNeighborsClassifier(n_neighbors=3)
x_train = train[['3P','BLK','TRB']]
y_train = train[['Pos']]

knn.fit(x_train,y_train.values.ravel())

x_test = test[['3P','BLK','TRB']]
y_test = test[['Pos']]

pred = knn.predict(x_test)
print('예측 결과: ', pred)

print(y_test.values) # 2차원 데이터
print(y_test.values.ravel()) # 1차원 데이터

from sklearn.metrics import accuracy_score
# accuracy_score() : confusion matrix(혼동행렬) 활용 정확도 계산
print('accuracy: ' +str( accuracy_score(y_test.values.ravel(), pred)))
print('acc: ', knn.score(x_test, y_test))

comparison = pd.DataFrame({'prediction':pred, 'truth value':y_test.values.ravel() })
print(comparison.head())