# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:07:56 2025

@author: migsh
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import optoanalysis

# Load data
data1 = optoanalysis.load_data("C:/Users/migsh/Desktop/Zicheng's data for reference/C2--3Sepdata--00019.trc")
time_data1, voltage1 = data1.get_time_data()
Gru = voltage1

# Define parameters
fs = 1000  # Sampling frequency (Hz)
target_freq = 13.0  # Target frequency from PSD
bandwidth = 1.2  # Width of the bandpass
# cutoff frequencies
lowcut = 12.7 # Lower cutoff freq
highcut = 13.5 # Upper cutoff freq

# Try different filter orders
orders = [2, 3, 4, 5]
filtered_signals = []

# make the butterworth bandpass
def butter_bandpass(lowcut, highcut, fs, order=4):
   
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
#Apply bandpass
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    # Ensure voltage data is 1D numpy array
    data = np.asarray(Gru).flatten()
    
    # Design and apply filter
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = signal.filtfilt(b, a, data)
    return y

# Create time array for plotting
t = np.arange(len(Gru)) / fs  # Time array in seconds

# Plot filter frequency response in figure
figure1 = plt.figure(figsize=(10, 6))
plt.title('Bandpass Filter Frequency Response')
for i, order in enumerate(orders):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    w, h = signal.freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label=f'Filter Response (Order={order})')
plt.xlim(6,20)
#plt.plot([0, 0.5*fs], [np.sqrt(0.5), np.sqrt(0.5)], '--', label='Target (Freq) Hz')
plt.axvline(x=lowcut, color='r', linestyle='--', label=f'Lowcut ({lowcut:} Hz)')
plt.axvline(x=highcut, color='r', linestyle='--', label=f'Highcut ({highcut:} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
