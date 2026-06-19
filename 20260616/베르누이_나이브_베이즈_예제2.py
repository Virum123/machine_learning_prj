import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import re

spam_df = pd.read_csv('/home/kk30100/deeplearning_prj/20260616/spam.csv')
# print(spam_df[:100])
# spam_100_df = spam_df[:100] 
# print(spam_100_df)

# df = re.findall(r'[A-Z a-z]', )
# email_list = []
# for i in len(spam_100_df):
#     a = spam_100_df[i]
#     b= re.findall( r'[A-Z a-z]', a )
#     email_list.append(b)

# print(email_list)

spamdf_subset = spam_df[:100].copy()

def EmailMessageControl(arg):
    return re.sub(r'[^a-zA-Z\s]','',arg)

spamdf_subset['Message'] = spamdf_subset['Message'].apply(EmailMessageControl)
print(spamdf_subset.info())
print(spamdf_subset)

spamdf_subset['Category'] = spamdf_subset['Category'].map({'ham':0, 'spam':1})
print(spamdf_subset)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

cv = CountVectorizer(binary=True)

train_x = cv.fit_transform(spamdf_subset['Message'])

np.set_printoptions(threshold=np.inf)

train_x_encoded = train_x.toarray()
print(train_x)

print(len( cv.get_feature_names_out() ) ) # 변환이 잘 됐는지, 값이 다 있는지 확인해봐야함

train_y = spamdf_subset['Category']

bnb = BernoulliNB()
train_y = train_y.astype('int')

bnb.fit(train_x_encoded, train_y)

print(bnb.score(train_x_encoded, train_y))

# 새로운 이메일 데이터 하나 추가 예측

temp_mail_cv = cv.transform(['last discount event of today',
                             'the payment document is attached and sent',
                             'company collaboration event free offer'])
print(temp_mail_cv.toarray())
predicted = bnb.predict(temp_mail_cv)
print(predicted)
print(len(predicted))
# for i in len(predicted):
#     if predicted[i] == 0:
#         predicted[i] = 'ham'
#     elif predicted[i] == 1:
#         predicted[i] ='spam'

print(predicted)