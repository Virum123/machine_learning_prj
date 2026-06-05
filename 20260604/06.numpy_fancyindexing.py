import numpy as np
import matplotlib.pyplot as plt

# range()
arr = np.arange(5,19).reshape(7,2)
print(arr)
# 행단위로 나누고, 짝수행만 data에 들어가도록하면 되겠네?
xdata = []
ydata = []
for i in range(len(arr)):
    if i % 2 != 0: 
        xdata.append(arr[ i, 0])
        ydata.append(arr[ i, 1])

print(xdata) # 7 11 15
print(ydata) # 8 12 16
# fancyindexing = 추출 위치의 인덱스를 배열 형태로 전달해서 추출
xdata = arr[ [1, 3, 5], 0]
ydata = arr[ [1, 3, 5], 1]
print(xdata) # 7 11 15
print(ydata) # 8 12 16
plt.scatter(arr[ [1, 3, 5], 0], arr[ [1, 3, 5], 1])
plt.savefig('fancyindexing.jpeg')