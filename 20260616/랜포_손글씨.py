from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
mnist = load_digits()
print(mnist['data'][-3:])
print(mnist['target'][-3:])
print(mnist) # 데이터랑 타겟 쓰는건 어떻게 아나요? 보통 자주 쓰는 이름을 넣어(data, target), 그리고 
features = mnist['data'] # 1797개의 8*8 이미지 데이터셋
labels = mnist['target'] #

RFmodel=RandomForestClassifier()
RFmodel.fit(features,labels)

predicted = RFmodel.predict(features[-5:])
print('labels: ', labels[-5:])
print('pre: ', predicted)

tempdata = [ 0. , 0. ,10. ,14. , 8. , 1. , 0. ,0. ,0. ,2. ,16. ,14. ,6. ,15. , 6. ,0. , 0. , 0.,
            12. ,15. , 8. ,15. , 0. , 0. , 0. , 15. ,5. ,16. ,16. ,10. ,0. ,0. ,0. ,0. ,12. ,15.,
            13. ,12. , 0. , 0. , 0. , 4. ,16. , 5. ,4. ,16. ,6. ,0. ,0. ,8. ,16. ,10. ,8. ,16.,
            8. , 1. , 0. , 1. , 7. ,12. ,14. ,12. ,1. ,0.]

import numpy as np
import matplotlib.pyplot as plt

temparr = np.array(tempdata)  # reshape
print(temparr)
temp_pred = RFmodel.predict([temparr])

print('temp_pred: ', temp_pred)

plt.imshow(temparr.reshape(8,8))
plt.savefig('랑포손')