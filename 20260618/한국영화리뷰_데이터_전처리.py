import numpy as np
import pandas as pd

reviewdf = pd.read_csv('/home/kk30100/deeplearning_prj/20260618/ratings_train.csv'
                        , header = 0, delimiter='\t', quoting=3)
print(reviewdf)
reviewdf.info()
reviewdf.dropna(how='any', inplace=True)
reviewdf.info()
print(reviewdf.head())
# 라벨(타겟) 컬럼데이터의 type을 실수에서 정수로 변환
reviewdf['label'] = reviewdf['label'].astype('int64')
reviewdf.info()
print(reviewdf.head())

# 리뷰 데이터의 항목중 중복 데이터가 있으면 찾아서 제거.
print(reviewdf['document'].nunique()) # n unique() ==> 유니크한 항목의 개수를 반환

reviewdf.drop_duplicates(subset='document', inplace=True)
reviewdf.info()
print(reviewdf.head())
# 결측치 및 중복이 제거된 총 데이터의 개수는 32,163개


import re
def reviewfiltering(arg):
    return re.sub(r'[^ㄱ-힣\s]','',arg)

# 한글과 공백을 제외한 모든 문자를 제거
# reviewdf['document'] = reviewdf['document'].str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '',regex=True) # series = 정규표현식 안먹을 수 있음, str속성으로 들어가서 .replace

reviewdf['document'] = reviewdf['document'].apply(reviewfiltering)

# def MultiSpacefiltering(arg):
#     return re.sub(r'^\s+','',arg)

# reviewdf['document'] = reviewdf['document'].apply(MultiSpacefiltering)

# def Nullreplace(arg):
#     return re.sub(r' ', np.nan ,arg)

# reviewdf['document'] = reviewdf['document'].apply(Nullreplace)

print(reviewdf.sample(100))

from konlpy.tag import Okt # konlpy version 0.6.0 설치
from tqdm import tqdm # 진행바 출력 # tqdm 4.62.0 version 설치
okt = Okt() # KoNLPy 제공 형태소 분석기
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를', '줄', '으로','자','에','와','한','하다']
# stem = True : 어간 추출 수행, 예) '이런' => '이렇다'로 변환

X_train = []
for sentence in tqdm(reviewdf['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 각 문장을 토큰화
    sentence_removed_stopwords = \
    [word for word in tokenized_sentence if not word in stopwords] 
    # 불용어제거#불용어 제거된 단어 리스트를 한 문장으로 합친 다음 X_train list 에 추가
    X_train.append(' '.join(sentence_removed_stopwords))

# print(X_train[:5]) # 불용어가 제거된 문장 모음 리스트
# print('='*80)
# print(X_train[:5])

reviewdf['document'] = X_train
print(reviewdf)