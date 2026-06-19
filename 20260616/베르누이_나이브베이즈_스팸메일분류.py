import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# 스팸 메일 분류를 위한 이메일 제목 & 스팸 레이블 준비

email_list = [
    {'email title':'free game only today', 'spam':True},
    {'email title': 'cheapest flight deal', 'spam':True}, 
    {'email title': 'limited time offer only today only today','spam':True}, 
    {'email title': 'today meeting schedule', 'spam':False}, 
    {'email title': 'your flight schedule attached', 'spam':False}, 
    {'email title': 'your credit card statement', 'spam':False}
]

email_df = pd.DataFrame(email_list)
print(email_df)

# 분류를 위해 label을 수치로 변환
email_df['spam'] = email_df['spam'].map({True:1, False:0})
print(email_df)

train_x = email_df['email title']
train_y = email_df['spam']

cv = CountVectorizer(binary=True)
train_x_cv = cv.fit_transform(train_x)
print(train_x_cv)
train_encoded = train_x_cv.toarray()
print(train_encoded)


# 모델 준비
bnb = BernoulliNB()
print(type(train_y))

# 모델 학습
bnb.fit(train_encoded, train_y)

print('acc: ', bnb.score(train_encoded, train_y))

# 임의의 메일 제목 예측
temp_mail_cv = cv.transform(['last discount event of today',
                             'the payment document is attached and sent',
                             'company collaboration event free offer'])
print(temp_mail_cv.toarray())
predicted = bnb.predict(temp_mail_cv)
print(predicted)