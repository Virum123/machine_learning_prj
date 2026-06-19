import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
np.set_printoptions    
Moviedf = pd.read_csv('/home/kk30100/deeplearning_prj/20260616/IMDB Dataset.csv')
print(Moviedf)
# Moviedf = Moviedf[:100].copy()
# sentiment 컬럼 라벨을 수치 데이터로 변경


def EmailMessageControl(arg):
    return re.sub(r'[^a-zA-Z\s]','',arg) # ^을 사용해서 아닌것이라고 표현함

Moviedf['review'] = Moviedf['review'].apply(EmailMessageControl)

Moviedf['sentiment'] = Moviedf['sentiment'].map({'positive':1, 'negative':0})

train_x = Moviedf['review']
train_y = Moviedf['sentiment']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
train_cv = cv.fit_transform(train_x)
# train_cv_encoded = train_cv.toarray()
# print(train_cv_encoded[0])
# print(train_cv.toarray)


from sklearn.naive_bayes import MultinomialNB # 다항분포 나이브베이즈
mnb = MultinomialNB()
mnb.fit(train_cv, train_y)

print('acc : ', mnb.score(train_cv, train_y))

# 새로운 영화 리뷰 데이터 입력해서 예측

new_review_cv = cv.transform([    'This movie was amazing and I really enjoyed it',
    'The story was boring and the acting was terrible',
    'I will never watch this movie again',
    'Great acting and wonderful story',
    'This film wasted my time'])

print(new_review_cv.toarray())
predicted = mnb.predict(new_review_cv)
print(predicted)

label_name = np.array(['negative', 'positive'])
print(label_name[predicted])