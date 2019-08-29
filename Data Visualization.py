import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# five=pd.read_csv("/home/hashini/Documents/RSSI0820Window/MeanRSSIData_W5.csv")
# six=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W6.csv")
# eight=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W8.csv")
# ten=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W10.csv")
# twelve=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W12.csv")
# fourteen=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W14.csv")
# sixteen=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W16.csv")
# eighteen=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W18.csv")
# twenty=pd.read_csv("//home/hashini/Documents/RSSI0820Window/MeanRSSIData_W20.csv")
# one_meter=pd.read_csv("/home/hashini/Documents/RSSI_Data/1.csv")
# two_meter=pd.read_csv("/home/hashini/Documents/RSSI_Data/2.csv")
# r1=half_meter["rssi"]
# r1=r1[0:53]
# r2=one_meter["rssi"][0:53]
# r3=two_meter["rssi"][0:53]
# r1=five["rssi"]
# r2=six["rssi"]
# r3=eight["rssi"]
# r4=ten["rssi"]
# r5=twelve["rssi"]
# r6=fourteen["rssi"]
# r7=sixteen["rssi"]
# r8=eighteen["rssi"]
# r9=twenty["rssi"]

# r1=five["power"]
# r2=six["power"]
# r3=eight["power"]
# r4=ten["power"]
# r5=twelve["power"]
# r6=fourteen["power"]
# r7=sixteen["power"]
# r8=eighteen["power"]
# r9=twenty["power"]
#
# r1=r1[5:]
# r2=r2[6:]
# r3=r3[8:]
# r4=r4[10:]
# r5=r5[12:]
# r6=r6[14:]
# r7=r7[16:]
# r8=r8[18:]
# r9=r9[20:]
#
# minv=min(len(r1),len(r2),len(r3),len(r4),len(r5),len(r6),len(r7),len(r8),len(r9))
#
# r1=r1[:minv]
# r2=r2[:minv]
# r3=r3[:minv]
# r4=r4[:minv]
# r5=r5[:minv]
# r6=r6[:minv]
# r7=r7[:minv]
# r8=r8[:minv]
# r9=r9[:minv]

# print("Mean_0",np.mean(r1))
# print("Mean_0.5",np.mean(r2))
# print("Mean_1",np.mean(r3))
# print("Mean_1.5",np.mean(r4))
# print("Mean_2",np.mean(r5))
# print("Mean_2.5",np.mean(r6))
# print("Mean_3",np.mean(r7))

# x1=np.arange(len(r1))
# plt.figure()
# plt.xlabel('Data Points')
# plt.ylabel('Recieved Signal Power(mW)')
# plt.legend()
# plt.plot(x1, r1,'r-',x1,r4,'b-',x1,r9,'g-')
# plt.show()
# x1=np.arange(len(r1))
# plt.figure()
# plt.xlabel('Data Points')
# plt.ylabel('RSSI(dBm)')
# plt.legend()
# plt.plot(x1, r1,'r-',x1,r4,'b-',x1,r9,'g-')
# plt.show()
win_size = 10
file1 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/40Raw.csv")
file2 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/41Raw.csv")
file3 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/43Raw.csv")
file4 = pd.read_csv("/home/hashini/Documents/WalkingRSSI/moving data 0829/44Raw.csv")
fig,ax=plt.subplots()
ax.scatter((file4["raw_rssi"])[win_size:],(file4["standard_deviation"])[win_size:])
# for i in range(len(file1["raw_rssi"])-win_size):
#     ax.scatter(((file1["raw_rssi"])[win_size:])[i], ((file1["standard_deviation"])[win_size:])[i],color=colors[iris['class'][i]])
ax.set_xlabel("raw_rssi")
ax.set_ylabel("standard_deviation")
plt.show()
# file = [file1, file2, file3, file4]
# raw_rssi = []
# mean_rssi = []
# SD = []
# for i in range(len(file)):
#     raw_rssi.append((file[i])["raw_rssi"])
#     mean_rssi.append((file[i])["mean_rssi"])
#     SD.append((file[i])["standard_deviation"])
#

# raw_rssi_1=file1["raw_rssi"]
# mean_rssi_1=file1["mean_rssi"]
# SD=file["standard_deviation"]
# for j in range(len(raw_rssi)):
#     x = np.arange(len(mean_rssi[j])-win_size)
#     fig, ax = plt.subplots()
#     # ax.plot(x, (raw_rssi[j])[win_size:], 'r', label="Raw RSSI")
#     # ax.plot(x, (mean_rssi[j])[win_size:], 'b', label="Mean RSSI")
#     ax.plot(x, (SD[j])[win_size:],'g',label="Standard Deviation")
#     plt.show()
# # ax.plot(x,SD[win_size:],'g',label="Standard Deviation")
# legend = ax.legend(loc='lower right', shadow=True, fontsize='small')

# plt.plot(x,raw_rssi[win_size:],'r',x,mean_rssi[win_size:],'b',x,SD[win_size:],'g')
# # handles, labels = ax.get_legend_handles_labels()
# # ax.legend(handles, labels)

# plt.show()
# plt.figure()
# plt.plot(x,SD[win_size:])
# plt.show()


