#iris is a bunch of data from a flower given by the developers to use as example (150 data)

from sklearn import tree 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import tensorflow as tf
from tensorflow.contrib import learn


#Oranges and apples
"""

#in features smooth is 1 and bumpy is 0
#in labels 1 is orange and 0 is apple

#DecisionTree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]


wg = input("weight? ")
bmp = input("bumpy or smooth? ")

if (bmp == "bumpy"):
    bmp = 0
elif (bmp == "smooth"):
    bmp =1

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

res = (clf.predict([[wg, bmp]]))

if (res == 1):
    print("orange")
else:
    print("apple")

"""

#flowers

"""
iris = load_iris()
#print(iris.feature_names)
#print(iris.target_names)
#print(iris.data[0])
#for i in range(len(iris.target)):
#    print("Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

#remove some data
test_idx = [0,50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

#viz code
"""

#dogs
"""
greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height, lab_height], 10, stacked = True, ec= "black", lw = 0.5, alpha = 0.5, color = ['r', 'b'])
plt.show()
"""

#An own classifier 
"""
#classifier = tree.DecisionTreeClassifier()
#Another classifier who looks to nearest neighbors
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()

#Create an own classifier
#Random classifier 
#Nearest neighbor classifier
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            #label = random.choice(self.y_train)
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train[0])):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


#K is number of neighbors to consider
#ecludian distance calculates the distance regardless the dimensions

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)
    


#Pipeline

iris = load_iris()

X = iris.data
y = iris.target

#test_size is size of the data used in test, in this case 75 values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


classifier = ScrappyKNN()

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

print(predictions)

print(accuracy_score(y_test, predictions))
"""

#TensorFlow for poets

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos