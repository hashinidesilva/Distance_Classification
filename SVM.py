import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def normalize(val):
    return (val-stat["mean"])/stat["std"]


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
train_dataset.pop("median_rssi")
test_dataset.pop("median_rssi")
train_dataset.pop("raw_rssi")
test_dataset.pop("raw_rssi")
train_dataset.pop("kalman filtered")
test_dataset.pop("kalman filtered")


print("No of Training data : ", len(train_dataset))
print("No of Test data : ", len(test_dataset))
print(test_labels)
stat=(train_dataset.describe()).transpose()

train_dataset = normalize(train_dataset)
test_dataset = normalize(test_dataset)
# print(train_dataset["standard_deviation"])
# print(test_labels)

svclassifier = SVC(kernel='linear')
clf=svclassifier.fit(train_dataset, train_labels) # train the algorithm on the training data
print(svclassifier.fit(train_dataset, train_labels).score(train_dataset,train_labels))
# val1,val2=normalize([-80,5])

y_pred = svclassifier.predict(test_dataset)
y_pred=y_pred.tolist()
test_labels=test_labels.tolist()
print("Actual Label  Predicted Label")
count=0
for i in range(len(y_pred)):
    print("  ",test_labels[i],"       ",y_pred[i])
    if ((test_labels[i]==1) and (y_pred[i]==1)):
        count+=1
act_count=test_labels.count(1)
acc=(count/act_count)
print('Accuracy = ',acc)
print("\nConfusion Matrix:")
matrix=confusion_matrix(test_labels,y_pred)
print(confusion_matrix(test_labels,y_pred))
print("\nClassification Report:")
print(classification_report(test_labels,y_pred))

