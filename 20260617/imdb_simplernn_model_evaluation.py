from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import re
(train_input, train_traget), (test_input, test_target) = imdb.load_data(num_words=500)

from tensorflow.keras.preprocessing.sequence import pad_sequences
test_seq = pad_sequences(test_input, maxlen=100)
print(test_seq.shape) # 
print(test_seq[0])

# 단어 집합 개수(500), 문장 길이 패딩(100)으로 설정되어 정확도가 다소 떨어짐
model = load_model('/home/kk30100/deeplearning_prj/best-simlernn-model.keras') # 앞에서 저장한 모델 복원
# test_seq 데이터셋으로 정확도 측정
print('정확도: %4f'%model.evaluate(test_seq, test_target)[1])

print(model.predict(test_seq[0:1]))
print(test_target[0])

# imdb.load_data()함수활용, num, words개 만큼의 단어집합으로 리뷰 문장을
# 토큰화 진행할 경우 0(패딩), 1(문장시작), 2(untoken), 3(unused)은 특별 토큰으로 
# 취급 및 추가 함으로 'this' 단어는 11 + 3 ==> 14 값으로 토큰화됨
# ==> IMDB 리뷰 데이터셋에서 정한 규칙임
# 실제 빈도수가 가장 높은 'the': 1 ==> 1+3 ==> 4값으로 토큰화됨
# **결론**
# word_to index= imdb.get_word_index() 활용한 임의의 문장 데이터
# imdb.load_data()로 토큰화 완료된 데이터로 훈련했으니
# 신규 예측에 있어서도 동일 방법으로 토큰화 진행한 데이터로 예측 진행해야 함

# word_to index ==> { 단어: 정수, 단어: 정수, ...}, 정수 1부터 매핑
word_to_index = imdb.get_word_index() # <===imdb 인덱스 매핑 사전반환
for key, value in word_to_index.items():
    if value ==1:
        print('key, value:', key, value) # key, value: the 1
print(word_to_index['this'])

negative_review_str = """
This was probably one of the worst movies I have watched in a long time.
The dialogue between the main characters is awkward, forced, and often unintentionally funny.
The plot barely makes sense, with random decisions replacing actual storytelling.
Even the actors seem lost, and their performances cannot save the material.
I would not recommend this unless someone wants to waste two hours and leave annoyed.
""" 
positive_review_str = """
This is not just a beautifully crafted movie; it is a deeply memorable experience.
The dialogue feels natural, sharp, and full of small details that reveal character.
The story is carefully paced, with each scene adding something meaningful.
The performances are excellent, especially from the lead actors who carry the film with restraint.
For me, this is easily a ten out of ten and a movie I would recommend without hesitation.
"""

def new_sentence_tokenization(sentence_arg): # 임의의 문장을 정수 데이터로 인코딩(토큰화)
    # 정규화를 이용한 문장 정리
    # 숫자, 알파벳, 공백 문자를 제외한 모든 문자를 ''로 치환(=제거), 이후 소문자화
    new_sentence = re.sub('[^0-9a-zA-z\s]','',sentence_arg).lower()
    # 정수 인코딩
    encoded =[ ]
    for word in new_sentence.split(' '):
        # eksdj 단어 집합 크기를 훈련 데이터와 동일하게 500으로 제한
        try:
            if word_to_index[word] <= 500:
                encoded.append( word_to_index[word]+3) # 예) 'the'의 value 값 1에 3을 더해 4를 저장
            else:
                #500 이상의 숫자는 <untoken> 알 수 없는 토큰으로 취급
                encoded.append(2)
        # 단어 집합에 없는 단어, 즉 word_to_index 단어 사전에 word 키 값이 없는 경우
        # <untoken> 알 수 없는 토큰으로 취급
        except KeyError:
            encoded.append(2)
    
    #pad_sequences(): 길이를 맞춰주는 패딩 진행(잘라내기 또는 0으로 채워짐)
    pad_new = pad_sequences( [encoded], maxlen = 100) # 타임스탬프 형성을 위한 2차원 배열 형태 전달
    # 훈련, 테스트 데이터와 동일하게 길이를 100으로 패딩( 타임스탬프 크기)
    #예측
    print(pad_new)
    score = float(model.predict(pad_new))
    print('score: ', score)
    if(score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰".format(score*100))
    else:
        print("{:.2f}% 확률로 부정 리뷰".format(1-score*100))

new_sentence_tokenization(negative_review_str) # 함수 호출
new_sentence_tokenization(positive_review_str) # 함수 호출