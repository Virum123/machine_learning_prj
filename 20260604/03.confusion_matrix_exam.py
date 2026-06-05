from sklearn.metrics import classification_report, confusion_matrix

y_true = [0, 1, 1, 1, 0, 1, 5, 6, 5, 5, 6, 5, 6]
y_pred = [0, 0, 0, 1, 0, 1, 5, 6, 5, 6, 6, 1, 1]    

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
# 다중과 이진을 구분하는 이유
# 개념은 비슷한데 계산 알고리즘이 다름 Y/N랑 이거 맞는거야? 랑 다른 그런 원리인 것 같다.

# 답안 비율이 불균형할 떄 F1 score를 이용한다.
# F1 score는 precision과 recall의 조화 평균으로, 불균형한 데이터셋에서 모델의 성능을 평가하는 데 유용함
# 요즘엔 잘 사용하지 않는다. 보통 균형을 맞추기 떄문

# scikit-learn은 잘 돌아가는지 확인하기 위한 패키지 정도로 생각하면 된다.
