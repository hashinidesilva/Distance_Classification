import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
file=pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/present.csv")
# print(len(file))
# file=file[10:]
# print(len(file))
# print(file)
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

train_dataset = dataset.sample(frac=0.8,random_state=0)
# x_dataset = dataset.drop(train_dataset.index)
# test_dataset = x_dataset.sample(frac=0.5,random_state=0)
test_dataset=dataset.drop(train_dataset.index)
# x_train=train_dataset.pop("raw_rssi")
# y_train=train_dataset.pop('distance')
# x_test=test_dataset.pop("raw_rssi")
# y_test=test_dataset.pop('distance')
train_labels = train_dataset.pop('distance')
test_labels = test_dataset.pop('distance')
train_dataset.pop("time")
test_dataset.pop("time")
# train_dataset.pop("raw_rssi(mW)")
# test_dataset.pop("raw_rssi(mW)")
train_dataset.pop("raw_rssi")
test_dataset.pop("raw_rssi")
train_dataset.pop("median_rssi")
test_dataset.pop("median_rssi")
# train_dataset.pop("standard_deviation")
# test_dataset.pop("standard_deviation")
# train_dataset.pop("dist")
# test_dataset.pop("dist")

stat=(train_dataset.describe()).transpose()
print(stat)

def normalize(val):
    return (val-stat["mean"])/stat["std"]


train_dataset = normalize(train_dataset)
test_dataset = normalize(test_dataset)
# print(test_dataset.shape)
# # ###Normalize train data
# # train_stats = train_dataset.describe()
# # mean_train=train_stats["mean"]
# # std_train=train_stats['std']
# # x_train=(x_train-mean_train)/std_train
# #
# # ####Normalize test data
# # test_stats = x_test.describe()
# # # print(test_stats)
# # mean_test=test_stats["mean"]
# # std_test=test_stats['std']
# # x_test=(x_test-mean_test)/std_test
#
# # mean_train=np.mean(x_train)
# # std_train=np.std(x_train)
# # x_train=(x_train-mean_train)/std_train
#
# # mean_test=np.mean(x_test)
# # std_test=np.std(x_test)
# # x_test=(x_test-mean_test)/std_test
# # x_unknown=unknown.values
# # print(len(unknown))
# x_unknown=unknown.pop("raw_rssi")
# # print(unknown)
# #print(x_unknown)
model=keras.Sequential([keras.layers.Dense(4,input_shape=[2],activation=tf.nn.relu),
                        keras.layers.Dense(8,activation=tf.nn.relu),
                        # keras.layers.Dense(64,activation=tf.nn.relu),
                        keras.layers.Dense(1)
                        ])
model.compile(optimizer='adam',loss="mean_squared_error",metrics=['mean_absolute_error', 'mean_squared_error'])
history=model.fit(train_dataset,train_labels,epochs=500,validation_split=0.2)
loss,mean_absolute_error,mean_squared_error=model.evaluate(test_dataset,test_labels)
print("Test Loss",mean_squared_error)
d1= model.predict(test_dataset)
# print(model.predict(np.array([[2.44,1.65]],dtype='float')))
# print(d1)
# d2=test_dataset.pop("distance")
# d2=d2.tolist()
d2=test_labels.tolist()
print("Predicted Distance   Actual Distance")
for i in range(len(d1)):
    print("    ",d1[i][0],"       ",d2[i])
# # print(d2)
# # # a1=pd.read_csv("/home/hashini/Documents/RSSI_Data/one_meter.csv")
# # # a2=pd.read_csv("/home/hashini/Documents/RSSI_Data/one_meter_withm.csv")
# # # r1=a1["rssi"]
# # # r2=a2["rssi_withMA"]
# # # x1=np.arange(len(r1))
# # # x2=np.arange(len(r2))
# # # # # plt.interactive(False)
# # # plt.figure()
# # # plt.plot(x1,r1)
# # # plt.show()
# # # #
# # # plt.figure()
# # # plt.plot(x2[10:],r2[10:])
# # # plt.show()
# # # #
# def plot_history(history):
#   hist = pd.DataFrame(history.history)
#   hist['epoch'] = history.epoch

  # plt.figure()
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean Abs Error [MPG]')
  # plt.plot(hist['epoch'], hist['mean_absolute_error'],
  #          label='Train Error')
  # plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
  #          label = 'Val Error')
  # # plt.ylim([0,5])
  # plt.legend()

  # plt.figure()
  # plt.xlabel('Epochs')
  # plt.ylabel('Mean Square Error($m^2$)')
  # plt.plot(hist['epoch'], hist['mean_squared_error'],
  #          label='Training Error')
  # plt.plot(hist['epoch'], hist['val_mean_squared_error'],
  #          label = 'Validation Error')
  # # plt.ylim([0,20])
  # plt.legend()
  # plt.savefig("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/regalln")
  # plt.show()
#
#
# plot_history(history)
# # # plt.scatter(d2, d1)
# # # plt.xlabel('True Values [MPG]')
# # # plt.ylabel('Predictions [MPG]')
# # # plt.axis('equal')
# # # plt.axis('square')
# # # plt.xlim([0,plt.xlim()[1]])
# # # plt.ylim([0,plt.ylim()[1]])
# # # _ = plt.plot([-100, 100], [-100, 100])
# #
# model.save("my_model.h5")
# converter=tf.lite.TFLiteConverter.from_keras_model_file("my_model.h5")
# tflite_model=converter.convert()
# open("converted.tflite","wb").write(tflite_model)