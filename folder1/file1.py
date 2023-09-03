import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

cma = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


iris = datasets.load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

#print(x_train.shape)
#print(x_train[0])
#print(y_train.shape)
#print(y_train)

#plt.figure()
#plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cma, edgecolor='k', s=20)
#plt.show()
import knn
from knn import KNN
class KNN:
    def _init_(self, k=3):
        self.k = k
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

clf = KNN(k=3)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
acc = np.sum(predictions == y_test) /  len(y_test)
print(acc)
