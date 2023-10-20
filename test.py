import numpy as np
import matplotlib.pyplot as plt
import csv

# Đọc tệp CSV và lấy dữ liệu từ các cột
with open('training.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Tạo vector A và vector b từ dữ liệu
A = [float(row['Height']) for row in data]
b = [float(row['Weight']) for row in data]

# Chuyển danh sách A và b thành mảng numpy
A = np.array(A).reshape(-1, 1)
b = np.array(b).reshape(-1, 1)
print(A)
# Visualize data
plt.plot(A, b, 'ro')

# Create vector ones
ones = np.ones((A.shape[0], 1), dtype=np.int8)

# Add a column of ones to A
A = np.concatenate((A, ones), axis=1)

# Solve for x
x = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
print(x)

# Test data to draw
x0 = np.array([1, 200])
y0 = x0 * x[0][0] + x[1][0]
plt.plot(x0, y0)

# Test predict data
x_test = 12
y_test = x_test * x[0][0] + x[1][0]
print(y_test)

plt.show()
