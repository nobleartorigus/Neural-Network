#pandas used to manage data and read comma-separated values(csv) in columns
#matplotlib used to plot
#numpy agrega soporte de vectores, matrices y funciones matematicas
#seaborn statistical data visualization
#sklearn to machine learning
#sklearn StandardScaler to standarize data
#metrics to evaluation matrixes
#Sequential to do multi-layer perceptron
#Dense is a fully connected layer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


y_pred = [int]


# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

#Print info on white wine
#print(white.info())

#Print info on red wine
#print(red.info())

# First rows of `red` 
#print(red.head())

# Last rows of `white`
#print(white.tail())

# Take a sample of 5 rows of `red`
#print(red.sample(5))

# Describe `white`
print(white.describe())

# Double check for null values in `red`
#print(pd.isnull(red))

fig1, ax1 = plt.subplots(1,2, figsize=(10, 10))


ax1[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax1[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig1.subplots_adjust(left=0.2, right=1, bottom=0.2, top=0.7, hspace=0.05, wspace=1)
ax1[0].set_ylim([0, 1000])
ax1[0].set_xlabel("Alcohol in % Vol")
ax1[0].set_ylabel("Frequency")
ax1[1].set_xlabel("Alcohol in % Vol")
ax1[1].set_ylabel("Frequency")
ax1[0].legend(loc='best')
ax1[1].legend(loc='best')
fig1.suptitle("Distribution of Alcohol in % Vol")

# plt.show()

#print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
#print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))

fig2, ax2 = plt.subplots(1,2, figsize=(10, 10))

ax2[0].scatter(red['quality'], red["sulphates"], color="red", edgecolors="black")
ax2[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax2[0].set_title("Red Wine")
ax2[1].set_title("White Wine")
ax2[0].set_xlabel("Quality")
ax2[1].set_xlabel("Quality")
ax2[0].set_ylabel("Sulphates")
ax2[1].set_ylabel("Sulphates")
ax2[0].set_xlim([0,10])
ax2[1].set_xlim([0,10])
ax2[0].set_ylim([0,2.5])
ax2[1].set_ylim([0,2.5])
fig2.subplots_adjust(wspace=0.5)
fig2.suptitle("Wine Quality by Amount of Sulphates")

# plt.show()

fig3, ax3 = plt.subplots(1,2, figsize=(10, 10))


np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax3[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax3[1].scatter(whitex, whitey, c=whitecolors[i])
    
ax3[0].set_title("Red Wine")
ax3[1].set_title("White Wine")
ax3[0].set_xlim([0,1.7])
ax3[1].set_xlim([0,1.7])
ax3[0].set_ylim([5,15.5])
ax3[1].set_ylim([5,15.5])
ax3[0].set_xlabel("Volatile Acidity")
ax3[0].set_ylabel("Alcohol")
ax3[1].set_xlabel("Volatile Acidity")
ax3[1].set_ylabel("Alcohol") 
#ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax3[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
#fig.suptitle("Alcohol - Volatile Acidity")
fig3.subplots_adjust(top=0.85, wspace=0.7)

#plt.show()


fig4, ax4 = plt.subplots(1, figsize=(10, 10))

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

sns.set()

corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

#plt.show()

# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y= np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)

y_pred = model.predict(X_test)

print(np.round(y_pred[:5]))

"""array([[0],
       [1],
       [0],
       [0],
       [0]], dtype=int32)"""

print(y_test[:5])

#array([0, 1, 0, 0, 0])

score = model.evaluate(X_test, y_test, verbose = 1)
print(score)

#confusion matrix, which is a breakdown of predictions into a table showing correct predictions and the types of incorrect predictions made. Ideally, you will only see numbers in the diagonal, which means that all your predictions were correct!
print(confusion_matrix(y_test, y_pred))

#Precision is a measure of a classifier’s exactness.
print(precision_score(y_test, y_pred))

#Recall is a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
print(recall_score(y_test, y_pred))

#The F1 Score or F-score is a weighted average of precision and recall.
print(f1_score(y_test, y_pred))

#The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.
print(cohen_kappa_score(y_test, y_pred))




