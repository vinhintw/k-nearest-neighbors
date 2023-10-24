import numpy as np
import math
import csv


def calculate_distance(p1, p2):
    dimension = len(p1)
    distance = 0

    for i in range(dimension):
        distance += (p1[i] - p2[i]) * (p1[i] - p2[i])

    return math.sqrt(distance)


def get_k_neighbors(training_X, label_y, point, k):
    distances = []
    neighbors = []

    # calculate distance from point to everything in training_X
    for i in range(len(training_X)):
        distance = calculate_distance(training_X[i], point)
        distances.append(distance)

    # position of k smallest distance
    index = []

    # Get k closet points
    while len(neighbors) < k:
        i = 0
        min_distance = 999999
        min_idx = 0
        while i < len(distances):
            # Skip the nearest points that have been counted
            if i in index:
                i += 1
                continue

            # Update smallest distance and index
            if distances[i] <= min_distance:
                min_distance = distances[i]
                min_idx = i

            i += 1

        # Add min index so we skip it in the next iteration
        index.append(min_idx)
        neighbors.append(label_y[min_idx])

    return neighbors


def highest_votes(labels):
    labels_count = [0, 0, 0]
    for label in labels:
        labels_count[label] += 1

    max_count = max(labels_count)
    return labels_count.index(max_count)


def predict(training_X, label_y, point, k):
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
    return highest_votes(neighbors_labels)


def accuracy_score(predicts, labels):
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == labels[i]:
            correct_count += 1
    accuracy = correct_count / total
    return accuracy


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
y_train = target[:training_size].astype(int)
X_test = data[training_size:, :]
y_test = target[training_size:].astype(int)

k = 5
y_predict = []
for p in X_test:
    label = predict(X_train, y_train, p, k)
    y_predict.append(label)

# print(y_predict)
# print(y_test)

acc = accuracy_score(y_predict, y_test)
print("Accuracy: ", acc)
