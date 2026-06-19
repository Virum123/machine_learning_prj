from konlpy.tag import Okt
import os
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)
pd.set_option('max_colwidth', 1000)

# 한국어 불용어 제거 데이터 로딩
X_train_data = pd.read_csv('train_stopwords_reviews.csv',usecols=['document','label'])
X_test_data = pd.read_csv('test_stopwords_reviews.csv',usecols=['document','label'])

word_size = 11775
tokenizer = Tokenizer(word_size)
tokenizer.fit_on_texts(X_train_data['document'])
okt = Okt()
best_model = load_model('mb_model.keras')
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

def new_review_predict(review_string): 
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]','', review_string) # 한국어와 공백 이외의내용삭제
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어제거
    print(new_sentence) # ['영화', '굿', '잼']
    # [new_sentence] : 불용어 처리된 단어 리스트를 정수 인코딩 sequences 데이터 형성을# 위해 하나로 묶어서([ ]) 변환해 줘야함
    encoded = tokenizer.texts_to_sequences( [new_sentence] ) # 정수 인코딩
    print(encoded) # [[1, 363, 334]] 
    sentence_padding = pad_sequences(encoded, maxlen = 30) # 패딩 적용 동일 길이 Sequences 형성
    print(sentence_padding)
    #[[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # 0 0 0 0 0 0 0 0 0 1 363 334]] 
    score = float(best_model.predict(sentence_padding) ) # new_sentence 예측
    if(score > 0.5): 
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100)) 
    else: 
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))