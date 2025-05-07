# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:25:36 2024

@author: User
"""

#Should speak for themselves
import optoanalysis
import numpy as np
import matplotlib.pyplot as plt


# Just figure size
optoanalysis.properties["default_fig_size"] = (7, 7)

#Load .trc files from C1 or C3 (detector 1) and C2 or C4 (detector 2) for same time
#frame
detector1 = optoanalysis.load_data("C:/Users/User/OneDrive - Zero Carbon Engineering/Desktop/FDM Print file/C1--3Sepdata--00126.trc")
detector2 = optoanalysis.load_data("C:/Users/User/OneDrive - Zero Carbon Engineering/Desktop/FDM Print file/C2--3Sepdata--00126.trc")

#We need the voltage and time data from said files
time_detector1, voltage1 = detector1.get_time_data(timeStart=0, timeEnd=10000)
time_detector2, voltage2 = detector2.get_time_data(timeStart=0, timeEnd=10000)


# We need to get the data into an array, numpy and its arrays make dealing with
#so many data points more tolerable following Tim's initial advice 
data =  np.array(list(zip(voltage1, voltage2)))

#So what do we do with all we've put together?
fig, ax = plt.subplots(figsize=(6, 6))
ax.axis('equal')
sc = ax.scatter(voltage1, voltage2, c=time_detector1[:len(voltage1)], cmap='viridis', s=1, alpha=0.5)

plt.xlabel(" Detector 1 Voltage X")
plt.ylabel(" Detector 2 Voltage Y")
plt.title("Scatter Plot for 3 Sept files ending 00126")
cbar = plt.colorbar(sc, label="Time (s)")

plt.legend()
plt.show()