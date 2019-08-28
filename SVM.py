import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("/home/hashini/Documents/RSSI0820/Mean_all.csv")   #chnage your file name
# unknowndata = pd.read_csv("unknownData.csv")

X = dataset.pop('rssi')
X=X.values
# X=X.tolist()
X=X.reshape(-1,1)
y = dataset.pop('distance')

#divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
# print(type(X_train))

# 5. Training the Algorithm
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train) # train the algorithm on the training data
y_pred = svclassifier.predict(X_test)
y_test=y_test.tolist()
print(type(y_pred),type(y_test))
for i in range(len(y_pred)):
    print(y_pred[i],y_test[i])
# x_val=np.arange(len(y_pred))
# plt.figure()
# plt.plot(x_val,y_pred,'ro',x_val,y_test,'go')
# plt.show()

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

# unknown_pred = svclassifier.predict(unknowndata)
