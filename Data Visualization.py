import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# win_size = 10
# file1 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/30Raw.csv")
# file2 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/31Raw.csv")
# file3 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/41Raw.csv")
# file4 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/fulldata.csv")
file= pd.read_csv("/home/hashini/Documents/NEW DATA/18-12-19/moving_500msRaw.csv")
raw_rssi=file["raw_rssi"]
mean_rssi=file["mean_rssi"]
SD=file["standard_deviation"]
kalman=file["kalman filtered"]
median_rssi=file["median_rssi"]

x = np.arange(len(raw_rssi))
plt.figure()
plt.xlabel("Data points")
plt.ylabel("SD ")
plt.title(" 8 meter ")
a,=plt.plot(x,raw_rssi,label='raw_rssi')
# b,=plt.plot(x,mean_rssi,label='mean_rssi')
c,=plt.plot(x,median_rssi,label='median_rssi')
# d,=plt.plot(x,kalman,label='kalman')
# plt.plot(x,SD)
plt.legend(handles=[ a,c])
# plt.savefig("/home/hashini/Documents/NEW DATA/18-12-19/8m_SD.png")
plt.show()

# plt.xlabel("Data points")
# plt.ylabel("Raw_rssi(dBm)")
# plt.plot(x,raw_rssi[win_size:])
# plt.savefig("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/Raw_dBm_4.png")
# plt.show()
# plt.figure()
# plt.xlabel("Data points")
# plt.ylabel("Raw_rssi(mW)")
# plt.plot(x,raw_mW[win_size:])
# plt.savefig("/home/hashini/Documents/WalkingRSSI/moving data 0829/Correct/Raw_mW_4.png")
# plt.show()

#
















