import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file=pd.read_csv("/home/hashini/Documents/RSSI0820/Mean_all.csv")
dataset=file.copy()
# unknown=pd.read_csv("/home/hashini/Documents/RSSI_Data_stage1/unknown.csv")
train_dataset = dataset.sample(frac=0.6,random_state=0)
x_dataset = dataset.drop(train_dataset.index)
test_dataset = x_dataset.sample(frac=0.5,random_state=0)
unknown= dataset.drop(test_dataset.index)
x=train_dataset.pop("rssi")
y=train_dataset.pop('distance')
x_test=test_dataset.pop("rssi")
y_test=test_dataset.pop('distance')

####Normalize train data
train_stats = x.describe()
mean_train=train_stats["mean"]
std_train=train_stats['std']
x=(x-mean_train)/std_train

####Normalize test data
test_stats = x_test.describe()
mean_test=test_stats["mean"]
std_test=test_stats['std']
x_test=(x_test-mean_test)/std_test

# x_unknown=unknown.values
x_unknown=unknown.pop("rssi")
y_unknown=unknown["distance"].tolist()

# print(len(x_unknown),len(y_unknown))
# print(x_unknown)
# print(y_unknown)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4,input_shape=[1],activation=tf.nn.relu),
  tf.keras.layers.Dense(8, activation=tf.nn.relu),
  tf.keras.layers.Dense(8, activation=tf.nn.relu),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])

print(x_test)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(x, y, epochs=500)
test_loss, test_acc=model.evaluate(x_test, y_test)
print("Test Accuracy",test_acc)
d1= model.predict(x_unknown)
for i in range(len(d1)):
    predi=np.argmax(d1[i])
    label=y_unknown[i]
    print(predi,label)

# def plot_history(history):
#   hist = pd.DataFrame(history.history)
#   hist['epoch'] = history.epoch
#
#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean Abs Error [MPG]')
#   plt.plot(hist['epoch'], hist['mean_absolute_error'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
#            label = 'Val Error')
#   # plt.ylim([0,5])
#   plt.legend()
#
#   plt.figure()
#   plt.xlabel('Epoch')
#   plt.ylabel('Mean Square Error [$MPG^2$]')
#   plt.plot(hist['epoch'], hist['mean_squared_error'],
#            label='Train Error')
#   plt.plot(hist['epoch'], hist['val_mean_squared_error'],
#            label = 'Val Error')
#   # plt.ylim([0,20])
#   plt.legend()
#   plt.show()


# plot_history(history)
# plt.scatter(d2, d1)
# plt.xlabel('True Values [MPG]')
# plt.ylabel('Predictions [MPG]')
# plt.axis('equal')
# plt.axis('square')
# plt.xlim([0,plt.xlim()[1]])
# plt.ylim([0,plt.ylim()[1]])
# _ = plt.plot([-100, 100], [-100, 100])




