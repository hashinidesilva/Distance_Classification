import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
file=pd.read_csv("/home/hashini/Documents/RSSI0820/Mean_all.csv")
dataset=file.copy()
# unknown=pd.read_csv("/home/hashini/Documents/RSSI Test/RSSIData_0.5.csv")
# x_t=dataset.sample(frac=0.2,random_state=0)
# x=dataset["rssi"].tolist()
# y=dataset["distance"].tolist()
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
# print(x_t)
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)
train_dataset = dataset.sample(frac=0.6,random_state=0)
x_dataset = dataset.drop(train_dataset.index)
test_dataset = x_dataset.sample(frac=0.5,random_state=0)
unknown=x_dataset.drop(test_dataset.index)
x_train=train_dataset.pop("rssi")
y_train=train_dataset.pop('distance')
x_test=test_dataset.pop("rssi")
y_test=test_dataset.pop('distance')

###Normalize train data
train_stats = x_train.describe()
mean_train=train_stats["mean"]
std_train=train_stats['std']
x_train=(x_train-mean_train)/std_train

####Normalize test data
test_stats = x_test.describe()
# print(test_stats)
mean_test=test_stats["mean"]
std_test=test_stats['std']
x_test=(x_test-mean_test)/std_test

# mean_train=np.mean(x_train)
# std_train=np.std(x_train)
# x_train=(x_train-mean_train)/std_train

# mean_test=np.mean(x_test)
# std_test=np.std(x_test)
# x_test=(x_test-mean_test)/std_test
# x_unknown=unknown.values
print(len(unknown))
x_unknown=unknown.pop("rssi")
# print(unknown)
#print(x_unknown)
model=keras.Sequential([keras.layers.Dense(32,input_shape=[1],activation=tf.nn.relu),
                        keras.layers.Dense(64,activation=tf.nn.relu),
                        keras.layers.Dense(64,activation=tf.nn.relu),
                        keras.layers.Dense(1)
                        ])
model.compile(optimizer='adam',loss="mean_squared_error",metrics=['mean_absolute_error', 'mean_squared_error'])
history=model.fit(x_train,y_train,epochs=250,validation_split=0.2)
loss,mean_absolute_error,mean_squared_error=model.evaluate(x_test, y_test)
print("Test Loss",mean_squared_error)
d1= model.predict(x_unknown)
# print(d1)
d2=unknown.pop("distance")
d2=d2.tolist()
for i in range(len(d1)):
    print(d1[i],d2[i])
# print(d2)
# # a1=pd.read_csv("/home/hashini/Documents/RSSI_Data/one_meter.csv")
# # a2=pd.read_csv("/home/hashini/Documents/RSSI_Data/one_meter_withm.csv")
# # r1=a1["rssi"]
# # r2=a2["rssi_withMA"]
# # x1=np.arange(len(r1))
# # x2=np.arange(len(r2))
# # # # plt.interactive(False)
# # plt.figure()
# # plt.plot(x1,r1)
# # plt.show()
# # #
# # plt.figure()
# # plt.plot(x2[10:],r2[10:])
# # plt.show()
# # #
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  # plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  # plt.ylim([0,20])
  plt.legend()
  plt.show()
#
#
plot_history(history)
# # plt.scatter(d2, d1)
# # plt.xlabel('True Values [MPG]')
# # plt.ylabel('Predictions [MPG]')
# # plt.axis('equal')
# # plt.axis('square')
# # plt.xlim([0,plt.xlim()[1]])
# # plt.ylim([0,plt.ylim()[1]])
# # _ = plt.plot([-100, 100], [-100, 100])
#


