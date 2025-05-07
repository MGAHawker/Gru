# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 19:35:25 2025

@author: migsh
"""
import optoanalysis as opt
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from IPython import get_ipython



data1 = opt.load_data("C:/Users/migsh/Desktop/Zicheng's data for reference/C2--3Sepdata--00015.trc")
time_data1, voltage1 = data1.get_time_data()

# Take data array from opened trc file via opto
Gru = voltage1
# Filter Params
fs = 1000  # Sampling freq (Hz)
lowcut = 12.7  # Lower cutoff frequency (Hz)
highcut = 13.5  # Upper cutoff frequency (Hz)
order = 4  # Filter order - keep high order for sharp cutoff, won't go above 4.

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

# Print data information before filtering
print(f"Input data shape: {np.asarray(Gru).shape}")
print(f"Input data type: {type(Gru)}")

# Convert data to numpy array if needed
Gru = np.asarray(Gru).flatten()
print(f"Processed data shape: {Gru.shape}")

# Apply the filter
filtered_signal = butter_bandpass_filter(Gru, lowcut, highcut, fs, order)

# Create time array for plotting
t = np.arange(len(Gru)) / fs  # Time array in seconds

# Figure for plots
figure1 = plt.figure(figsize=(15, 10))

# Original signal plot
#plt.subplot(3, 1, 1)
plt.plot(t, Gru, 'b-', label='Original Signal')
plt.xlim(1100, 1105) #Shows a smaller frame of time so I can see if it looks better
plt.ylim(8.39, 8.48)
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.title('Original Signal')
plt.grid(True)
plt.legend()

# Filtered signal plot
figure2 = plt.figure(figsize=(15, 10))
#plt.subplot(3, 1, 2)
plt.plot(t, filtered_signal, 'g-', label=f'Filtered Signal: Lowcut({lowcut:} Hz)- Highcut({highcut:} Hz)')
plt.xlim(1100, 1105) #Shows a smaller frame of time so I can see if it looks better
plt.xlabel('Time (seconds)')
plt.ylabel('Voltage')
plt.title('Filtered Signal')
plt.grid(True)
plt.legend()

# Frequency spectrum plot
figure3= plt.figure(figsize=(30,30))
frequencies = np.fft.fftfreq(len(filtered_signal), 1/fs)
spectrum = np.abs(np.fft.fft(filtered_signal))
mask = (frequencies > 0) & (frequencies <= 20)
plt.plot(frequencies[mask], spectrum[mask], 'r-', label='Frequency Spectrum')
plt.axvline(x=13, color='k', linestyle='--', label='Target 13 Hz')
plt.xlim(12.5,13.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Filtered Signal')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()