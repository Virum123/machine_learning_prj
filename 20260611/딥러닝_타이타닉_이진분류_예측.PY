
from tensorflow.keras.models import load_model

titanic_bestmodel = load_model("titanic_bestmodel.keras")
titanic_bestmodel.summary()


# 머신러닝 처럼 새로운 3명의 정보를 만들어
# 생존여부 예측추론
import numpy as np
f1 = np.array([
              [1, 52, 1, 0],
              [0, 26, 1, 0],
              [0, 8, 0, 1],
              [1, 66, 0, 1]
              ])

pred = titanic_bestmodel.predict(f1)
# titanic_bestmodel.predict(f2)
# titanic_bestmodel.predict([0,8,0,1])
# titanic_bestmodel.predict([1,66,0,1])

print(pred)
