# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 11:39:03 2022

@author: cjt1g20
"""

import optoanalysis as oa
import scipy.signal
import matplotlib.pyplot as plt
import numpy as np
 

def get_dataoa():
    #data = oa.load_data('031122_maglev/20221103-0002.csv')
    #data = oa.load_data('180523_levitation\\20230519-0004.csv')
    #data = oa.load_data('2823_diamag\\20231002-0002.csv')
    data = oa.load_data("C:/Users/migsh/Desktop/moon325/C2--goodcalib--00001.trc",silent=True)
    voltages = data.voltage
    time_data = data.time.get_array()[0:len(voltages)]
    return time_data, voltages, data
    


tr, raw, data = get_dataoa()

freqx = 100.37
subsample_amount = 150 #play around with until filter works. 10 seems sensible often



tr = tr #+ 100.0
tx, x = data.filter_data(freqx, subsample_amount, PeakWidth = 0.5)[0:2]#filters frequency
#tx, x = data.filter_data(freqx, PeakWidth = 0.5)[0:2]
tr = tr[1:]
tx = tx #+ 100.0
fig, ax0 = plt.subplots()
ax0.plot(tr, raw[:len(tr)], alpha=0.4, label='raw')
ax0.plot(tx, x, alpha=0.4, label='x')
ax0.legend()#plots time data and filtered time data


Hx = scipy.signal.hilbert(x)
Htransx = np.sqrt(Hx.imag**2 + x**2)

fig, ax1 = plt.subplots()
ax1.plot(tx, x, alpha=0.4, label='x')
ax1.plot(tx, Htransx, alpha=0.4, label='Htransx')
ax1.legend()#plots filtered time data and amplitude of filtered data

#print(tx)
StartTime = -4.5 #57#56.0 #20
EndTime = StartTime + 3.50 #200
StartIndex = list(tx).index(tx[tx >= (StartTime )][0])
EndIndex = list(tx).index(tx[tx >= (EndTime )][0])#pick part of time trace to fit to
#print(tx)
print(StartIndex)
print(EndIndex)
t_slice = tx[StartIndex:EndIndex]
x_slice = x[StartIndex:EndIndex]
Htransx_slice = Htransx[StartIndex:EndIndex]
print("tsclie: "+ str(t_slice))
start_of_slice = t_slice[0] 

t_slice = t_slice - start_of_slice  # shift 0 point of time to start of slice

startOfDamping = max(Htransx_slice)
w = freqx*2*np.pi
Q = 400 # Q initial approx for fitting damping

fitfn = lambda t,a,b: a*np.exp(-b/2*t)#fits ringdown

result = scipy.optimize.curve_fit(fitfn,  t_slice,  Htransx_slice, p0=[startOfDamping, w/Q])
a, b = result[0]


print("damping = {:.5f} radians/s".format(b))#prints damping
print ("Q: "+str(w/b))#prints Q factor
print(a)
print(b)
fig, ax2 = plt.subplots(figsize = (10,5))
ax2.plot(t_slice, fitfn(t_slice, a, b), color='black')
ax2.plot(t_slice, x_slice, color = 'blue', alpha=0.2)
ax2.set_ylabel("Oscillation Amplitude (V)", fontsize = 15)
ax2.set_xlabel("Time (s)", fontsize = 15)
ax2.set_title("Ringdown of driven mode, start: "+str(StartTime)+"s, end: "+str(EndTime)+"s", fontsize = 15)
ax2.text(30,0.01,"Q: "+str(round(w/b)), fontsize = 15)
ax2.tick_params(axis='both', which='both', length = 5,labelsize = 15)
#ax2.set_xlim(0,40)
plt.tight_layout()#plots fit to filtered ringdown data 