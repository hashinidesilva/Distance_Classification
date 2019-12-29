import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
sns.set()

raw_dataset = pd.read_csv("/home/hashini/Documents/NEW DATA/moving_800msRaw.csv")   #chnage your file name
# raw_dataset=raw_dataset[10:]
dataset = raw_dataset.copy()
# unknowndata = pd.read_csv("unknownData.csv")

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('distance')
test_labels = test_dataset.pop('distance')
train_dataset.pop("time")
test_dataset.pop("time")
train_dataset.pop("mean_rssi")
test_dataset.pop("mean_rssi")
train_dataset.pop("raw_rssi")
test_dataset.pop("raw_rssi")
# train_dataset.pop("kalman filtered")
# test_dataset.pop("kalman filtered")
train_dataset.pop("median_rssi")
test_dataset.pop("median_rssi")

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(train_dataset,train_labels)
y_pred=knn.predict(test_dataset)
test_labels=test_labels.tolist()
print("Actual Label  Predicted Label")
count=0
for i in range(len(y_pred)):
    print("  ",test_labels[i],"       ",y_pred[i])
    if (test_labels[i]==1) and (y_pred[i]==1):
        count+=1
act_count=test_labels.count(1)
acc=(count/act_count)
print('Accuracy = ',acc)
print("\nConfusion Matrix:")
matrix=confusion_matrix(test_labels,y_pred)
print(confusion_matrix(test_labels,y_pred))
print("\nClassification Report:")
print(classification_report(test_labels,y_pred))

