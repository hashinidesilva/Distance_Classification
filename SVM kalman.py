import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt


# raw_dataset = pd.read_csv("/home/hashini/Documents/WalkingRSSI//moving data 0829/Correct/30Raw.csv")   #chnage your file name
raw_dataset = pd.read_csv("/home/hashini/Documents/Kalman_filtered_data/26-11-19/dis_moving_slow1Raw.csv")   #change your file name
# raw_dataset=raw_dataset[10:]
dataset = raw_dataset.copy()
# unknowndata = pd.read_csv("unknownData.csv")

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.pop('distance')
test_labels = test_dataset.pop('distance')
train_dataset.pop("time")
test_dataset.pop("time")
train_dataset.pop("median_rssi")
test_dataset.pop("median_rssi")
train_dataset.pop("raw_rssi")
test_dataset.pop("raw_rssi")
train_dataset.pop("mean_rssi")
test_dataset.pop("mean_rssi")
# print("No of Training data : ", len(train_dataset))
# print("No of Test data : ", len(test_dataset))
stat=(train_dataset.describe()).transpose()
# print(test_labels)

def normalize(val):
    return (val-stat["mean"])/stat["std"]

train_dataset = normalize(train_dataset)
test_dataset = normalize(test_dataset)

svclassifier = SVC(kernel='linear')
clf=svclassifier.fit(train_dataset, train_labels) # train the algorithm on the training data
print(svclassifier.fit(train_dataset, train_labels).score(train_dataset,train_labels))
# val1,val2=normalize([-80,5])

y_pred = svclassifier.predict(test_dataset)
test_labels=test_labels.tolist()
print("Actual Label  Predicted Label")
for i in range(len(y_pred)):
    print("  ",test_labels[i],"       ",y_pred[i])
print("\nConfusion Matrix:")
matrix=confusion_matrix(test_labels,y_pred)
print(confusion_matrix(test_labels,y_pred))
print("\nClassification Report:")
print(classification_report(test_labels,y_pred))
