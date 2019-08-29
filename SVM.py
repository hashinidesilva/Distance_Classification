import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
import matplotlib.pyplot as plt

raw_dataset = pd.read_csv("/home/hashini/Documents/WalkingRSSI/walking data third floor.csv")   #chnage your file name
# raw_dataset = pd.read_csv("/home/hashini/Documents/WalkingRSSI/test.csv")   #chnage your file name

dataset = raw_dataset.copy()
# unknowndata = pd.read_csv("unknownData.csv")

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('distance')
test_labels = test_dataset.pop('distance')
train_dataset.pop("time")
test_dataset.pop("time")
print("No of Training data : ", len(train_dataset))
print("No of Test data : ", len(test_dataset))
stat=(train_dataset.describe()).transpose()


def normalize(val):
    return (val-stat["mean"])/stat["std"]


train_dataset = normalize(train_dataset)
test_dataset = normalize(test_dataset)
# print(train_dataset)
print(test_labels)

svclassifier = SVC(kernel='rbf')
svclassifier.fit(train_dataset, train_labels) # train the algorithm on the training data
y_pred = svclassifier.predict(test_dataset)
print(type(y_pred))
print(type(test_labels))
test_labels=test_labels.tolist()
for i in range(len(y_pred)):
    print(test_labels[i],y_pred[i])
# print(y_pred,test_labels)
# print(test_labels)
# y_test=y_test.tolist()
# print(type(y_pred),type(y_test))
# for i in range(len(y_pred)):
#     print(y_pred[i],y_test[i])
# # x_val=np.arange(len(y_pred))
# # plt.figure()
# # plt.plot(x_val,y_pred,'ro',x_val,y_test,'go')
# # plt.show()
#
print("\nConfusion Matrix:")
matrix=confusion_matrix(test_labels,y_pred)
print(confusion_matrix(test_labels,y_pred))
print("\nClassification Report:")
print(classification_report(test_labels,y_pred))

fig = plt.figure()
ax = fig.add_subplot(111)

# plot the matrix
cax = ax.matshow(matrix)

# add colorbar for reference
fig.colorbar(cax)

# add labels to plot
plt.xlabel("Predicted")
plt.ylabel("True")
# plt.savefig("Conf_Mat.jpg")
plt.show()

# X = X.values
# # X=X.tolist()
# X = X.reshape(-1,1)
# y = dataset.pop('distance')

#divide data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# # print(type(X_train))
#
# # 5. Training the Algorithm
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train) # train the algorithm on the training data
# y_pred = svclassifier.predict(X_test)
# y_test=y_test.tolist()
# print(type(y_pred),type(y_test))
# for i in range(len(y_pred)):
#     print(y_pred[i],y_test[i])
# # x_val=np.arange(len(y_pred))
# # plt.figure()
# # plt.plot(x_val,y_pred,'ro',x_val,y_test,'go')
# # plt.show()
#
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test,y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test,y_pred))
#
# # unknown_pred = svclassifier.predict(unknowndata)
