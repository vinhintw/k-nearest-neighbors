from sklearn import neighbors
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

# Read CSV file
with open("data.csv", mode="r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    data = [row for row in reader]

# Create vector A, B from first column and second column
A = [float(row["Height"]) for row in data]
B = [float(row["Weight"]) for row in data]

# Create sex list(label) of vector A and B
sex = [str(row["Sex"]) for row in data]

# Convert vector to numpy array
A = np.array(A).reshape(-1, 1)
B = np.array(B).reshape(-1, 1)

# create data from A and B vector
data = np.hstack((A, B))
target = np.array(sex)
# Sex(label)  0 = Female ;  1 = Male

# training size
training_size = int((len(data) / 100) * 80)
# test size
test_size = len(data) - training_size

# shuffle data
randIndex = np.arange(data.shape[0])
np.random.shuffle(randIndex)
data = data[randIndex]
target = target[randIndex]

X_train = data[0:training_size, :]
y_train = target[:training_size]
X_test = data[training_size:, :]
y_test = target[training_size:]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

# init knn model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict
y_predict = knn.predict(X_test)
# show
print(y_predict)
print(y_test)

# Accuracy knn model
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)
