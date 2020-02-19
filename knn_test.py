from knn import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import operator

# Load data

digits = datasets.load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


# Inspect data

print(X_train.shape)
print(X_train[0])

print(y_train.shape)
print(y_train)


# Plot a digit

plt.figure()
plt.imshow(digits.images[0], cmap=plt.cm.get_cmap(
    'gray_r'), interpolation='nearest')
plt.show()


# Train KNN

k_score = {}
for k in range(2, 21):
    clf = KNN(k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    k_score[k] = clf.score(y_test, y_pred)
    print(f'accuracy for k={k} : {k_score[k]}')

best_k = max(k_score.items(), key=operator.itemgetter(1))[0]
print(f'Best k : {best_k}')
