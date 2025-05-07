# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:31:37 2025

@author: migsh
"""

from matplotlib.ticker import MultipleLocator
import optoanalysis
from frange import frange
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.signal
import seaborn as _sns
_sns.reset_orig()
from bisect import bisect_left as _bisect_left
import warnings as _warnings
from IPython import get_ipython
from scipy.optimize import curve_fit
import pandas as pd
import os
import glob

class DataObject():
    def __init__(self, filepath=None, voltage=None, time=None, SampleFreq=None, timeStart=None, RelativeChannelNo=None, PointsToLoad=-1, calcPSD=True, NPerSegmentPSD=1000000, NormaliseByMonitorOutput=False):
        self.filepath = filepath
        if self.filepath != None:
            self.filename = filepath.split("/")[-1]
            self.filedir = self.filepath[0:-len(self.filename)]
            self.load_time_data(RelativeChannelNo, SampleFreq, PointsToLoad, NormaliseByMonitorOutput)
        else:
            if voltage is not None:
                self.load_time_data_from_signal(voltage, time, SampleFreq, timeStart)
            else:
                raise ValueError("Must provide one of filepath or voltage to instantiate a DataObject instance")
        print("calcPSD: ", calcPSD)
        if calcPSD:
            print("running self.get_PSD")
            self.get_PSD(NPerSegmentPSD)
        return None
    def load_time_data(self, RelativeChannelNo=None, SampleFreq=None, PointsToLoad=-1, NormaliseByMonitorOutput=False):
           f = open(self.filepath, 'rb')
           raw = f.read()
           f.close()
           FileExtension = self.filepath.split('.')[-1]
           if FileExtension == "raw" or FileExtension == "trc":
               with _warnings.catch_warnings(): # supress missing data warning and raise a missing
                   # data warning from optoanalysis with the filepath
                   _warnings.simplefilter("ignore")
                   try:
                       waveDescription, timeParams, self.voltage, _, missingdata = optoanalysis.LeCroy.InterpretWaveform(raw, noTimeArray=True)
                   except IndexError as error:
                       print('problem with file {}'.format(self.filepath), flush=True)
                       raise(error)
               if missingdata:
                   _warnings.warn("Waveform not of expected length. File {} may be missing data.".format(self.filepath))
               self.SampleFreq = (1 / waveDescription["HORIZ_INTERVAL"])
               startTime, endTime, Timestep = timeParams
               self.timeStart = startTime
               self.timeEnd = endTime
               self.timeStep = Timestep
               self.time = frange(startTime, endTime+Timestep, Timestep)
               return None
    def get_PSD(self, NPerSegment=1000000, window="hann", timeStart=None, timeEnd=None, override=False):
            print("Calculating power spectral density")
            if timeStart == None and timeEnd == None:
                freqs, PSD = calc_PSD(self.voltage, self.SampleFreq, NPerSegment=NPerSegment)
                self.PSD = PSD
                self.freqs = freqs
            else:
                if timeStart == None:
                    timeStart = self.timeStart
                if timeEnd == None:
                    timeEnd = self.timeEnd

                time = self.time.get_array()

                StartIndex = _np.where(time == take_closest(time, timeStart))[0][0]
                EndIndex = _np.where(time == take_closest(time, timeEnd))[0][0]

                if EndIndex == len(time) - 1:
                    EndIndex = EndIndex + 1 # so that it does not remove the last element
                freqs, PSD = calc_PSD(self.voltage[StartIndex:EndIndex], self.SampleFreq, NPerSegment=NPerSegment)
                if override == True:
                    self.freqs = freqs
                    self.PSD = PSD

            return freqs, PSD
    def plot_improved_PSD(self, xlim=None, units="Hz", show_fig=True, timeStart=None, timeEnd=None, *args, **kwargs):
        if timeStart == None and timeEnd == None:
            freqs = self.freqs
            PSD = self.PSD
        else:
            freqs, PSD = self.get_PSD(timeStart=timeStart, timeEnd=timeEnd)

        unit_prefix = units[:-2]
        if xlim == None:
            xlim = [0, unit_conversion(self.SampleFreq/2, unit_prefix)]
        fig = _plt.figure(figsize=properties['default_fig_size'])
        ax = fig.add_subplot(111)
        ax.semilogy(unit_conversion(freqs, unit_prefix), PSD, *args, **kwargs)
        ax.set_xlabel("Frequency ({})".format(units))
        ax.set_xlim(xlim)
        #_plt.gca().yaxis.set_major_locator(MultipleLocator(1000))
        _plt.gca().xaxis.set_major_locator(MultipleLocator(10))
        _plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which='major', color='k', linestyle='-')
        ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
        ax.set_ylabel("$S_{xx}$ ($V^2/Hz$)")
        if show_fig == True:
            _plt.show()
        return fig, ax
    def analyse_drift(self, chunk_size=250, freq_range=None, plot_waterfall=True):
        """
        Analyse drift of peaks by dividing data into chunks and calculating PSD for each.
        Uses Lorentzian fit for each chunk for clear peak frequency used for drift analysis.
        Plots all chunk PSDs on the same graph with different colors.
        
        Parameters:
            -----------
            chunk_size : float
            Size of each chunk in seconds
            freq_range : tuple
            Optional (min_freq, max_freq) to focus on specific frequency range
            plot_waterfall : bool
            Whether to plot waterfall diagram of PSDs
            
            Returns:
                --------
                tuple
                (peak_freqs, peak_powers, time_centers)
                """
        print(f"Analyzing peak drift using {chunk_size} second chunks")
    
        # Convert frange time to numpy array if needed
        if hasattr(self.time, 'get_array'):
            # Use the get_array method if available
            time_array = self.time.get_array()
        else:
            # Otherwise try to convert to numpy array
            time_array = _np.array(list(self.time))
    
        voltage = self.voltage
    
        # Get time boundaries
        start_time = time_array[0] if len(time_array) > 0 else 0
        end_time = time_array[-1] if len(time_array) > 0 else 0
        total_duration = end_time - start_time
    
        # Calculate chunk parameters
        num_chunks = max(1, int(total_duration / chunk_size))
        time_centers = []
    
        # Set up plot for all PSDs on same graph
        combined_fig, combined_ax = _plt.subplots(figsize=(12, 8))
    
        # Create a colourmap for the chunks
        colours = _plt.cm.viridis(_np.linspace(0, 1, num_chunks))
    
        if plot_waterfall:
            waterfall_fig = _plt.figure(figsize=(12, 8))
            waterfall_ax = waterfall_fig.add_subplot(111, projection='3d')
    
        all_freqs = []
        all_psds = []
        peak_freqs = []
        peak_powers = []
        chunk_labels = []
    
        # Define frequency bands for z and x/y mode?
        mode1_range = (12, 12.75)  # Adjust based on your data
        mode2_range = (12.75, 13.25)    # Adjust based on your data
    
        # Create containers for tracking the fitted peaks
        mode1_peaks = []  # will store (time, freq, amplitude, width) tuples
        mode2_peaks = []  # will store (time, freq, amplitude, width) tuples
    
        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Calculate time boundaries chunk by chunk
            chunk_start_time = start_time + chunk_idx * chunk_size
            chunk_end_time = min(chunk_start_time + chunk_size, end_time)
            chunk_center = (chunk_start_time + chunk_end_time) / 2
            time_centers.append(chunk_center)
            chunk_labels.append(f"t={chunk_start_time:.1f}s to {chunk_end_time:.1f}s")
        
            # Get indices for data in chunk
            start_idx = _np.where(time_array >= chunk_start_time)[0]
            start_idx = start_idx[0] if len(start_idx) > 0 else 0
        
            end_idx = _np.where(time_array <= chunk_end_time)[0]
            end_idx = end_idx[-1] if len(end_idx) > 0 else len(time_array)-1
        
            # Extract chunk data
            chunk_time = time_array[start_idx:end_idx+1]
            chunk_voltage = voltage[start_idx:end_idx+1]
        
            # Calculate PSD for this chunk
            freqs, psd = calc_PSD(chunk_voltage, self.SampleFreq, NPerSegment=100000, window="hann")
            all_freqs.append(freqs)
            all_psds.append(psd)
        
            # Apply frequency range filter if specified
            if freq_range:
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                plot_freqs = freqs[mask]
                plot_psd = psd[mask]
            else:
                plot_freqs = freqs
                plot_psd = psd
        
            # Fit Lorentzian to mode 1 peak
            mode1_center, mode1_amplitude, mode1_width = fit_lorentzian_to_peaks(plot_freqs, plot_psd, mode1_range)
        
            # Fit Lorentzian to mode 2 peak
            mode2_center, mode2_amplitude, mode2_width = fit_lorentzian_to_peaks(plot_freqs, plot_psd, mode2_range)
        
            # Add fitted peaks to tracking lists
            if mode1_center is not None:
                mode1_peaks.append((chunk_center, mode1_center, mode1_amplitude, mode1_width))
            
            if mode2_center is not None:
                mode2_peaks.append((chunk_center, mode2_center, mode2_amplitude, mode2_width))
        
            # Plot PSD for this chunk on combined plot
            combined_ax.semilogy(plot_freqs, plot_psd, color=colours[chunk_idx], 
                            alpha=0.8, label=f"Chunk {chunk_idx+1}: {chunk_start_time:.1f}s-{chunk_end_time:.1f}s")
        
           # If we have Lorentzian fits, plot them too
            x_dense = _np.linspace(mode1_range[0], mode1_range[1], 1000)
            if mode1_center is not None:
               y_fit = lorentzian(x_dense, mode1_amplitude, mode1_center, mode1_width, 0)
               combined_ax.semilogy(x_dense, y_fit, '--', color=colours[chunk_idx], linewidth=1)
               combined_ax.plot([mode1_center], [mode1_amplitude], 'o', color=colours[chunk_idx])
            
            x_dense = _np.linspace(mode2_range[0], mode2_range[1], 1000)
            if mode2_center is not None:
                y_fit = lorentzian(x_dense, mode2_amplitude, mode2_center, mode2_width, 0)
                combined_ax.semilogy(x_dense, y_fit, '--', color=colours[chunk_idx], linewidth=1)
                combined_ax.plot([mode2_center], [mode2_amplitude], 'o', color=colours[chunk_idx])
    
        # Extract data for return and plotting
        mode1_times = [p[0] for p in mode1_peaks]
        mode1_freqs = [p[1] for p in mode1_peaks]
        mode1_amps = [p[2] for p in mode1_peaks]
        mode1_widths = [p[3] for p in mode1_peaks]

        mode2_times = [p[0] for p in mode2_peaks]
        mode2_freqs = [p[1] for p in mode2_peaks]
        mode2_amps = [p[2] for p in mode2_peaks]
        mode2_widths = [p[3] for p in mode2_peaks]
        # Create peak drift plot
        drift_fig, drift_ax = _plt.subplots(figsize=(12, 8))
        
        # Plot mode1 peaks
        if mode1_peaks:
            drift_ax.plot(mode1_times, mode1_freqs, 'o-', color='blue', 
                     label=f"{_np.mean(mode1_freqs):.2f} Hz")
                     
        # Plot mode2 peaks
        if mode2_peaks:
            drift_ax.plot(mode2_times, mode2_freqs, 'o-', color='cyan', 
                     label=f"{_np.mean(mode2_freqs):.2f} Hz")
        
        
        # Finalise drift plot
        drift_ax.set_xlabel('Time (s)')
        drift_ax.set_ylabel('Frequency (Hz)')
        drift_ax.set_title('Peak Frequency Drift Over Time (Lorentzian Fitted)')
        drift_ax.grid(True)
        drift_ax.legend()
        _plt.tight_layout()
       
        
        chunk_summary = {}
        for chunk_idx in range(num_chunks):
            chunk_start = start_time + chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, end_time)
            time_label = f"{chunk_start:.1f}s-{chunk_end:.1f}s"
       
            # Find frequencies for this chunk
            mode1_freq = "N/A"
            mode2_freq = "N/A"
        
            for i, t in enumerate(time_centers):
                if abs(t - (start_time + chunk_idx * chunk_size + chunk_size/2)) < 0.01:  # Close enough to this chunk's center
                    if chunk_idx < len(mode1_peaks) and chunk_idx < len(mode1_times):
                        mode1_freq = f"{mode1_freqs[chunk_idx]:.3f} Hz"
                    if chunk_idx < len(mode2_peaks) and chunk_idx < len(mode2_times):
                        mode2_freq = f"{mode2_freqs[chunk_idx]:.3f} Hz"
                    break
                
            chunk_summary[chunk_idx+1] = {
                "time_range": time_label,
                "mode1": mode1_freq,
                "mode2": mode2_freq
                }   
       
            
   
        # Create a figure for the table
        table_fig = _plt.figure(figsize=(10, len(chunk_summary) * 0.4 + 1))
        table_ax = table_fig.add_subplot(111)
        table_ax.axis('off')
   
        # Prepare data for table
        table_data = []
        for chunk, data in chunk_summary.items():
            table_data.append([chunk, data['mode1'], data['mode2']])
   
        # Create the table
        cols = ['Chunk', 'Z mode peak freq(Hz)', 'Mode 2 peak freq(Hz)']
        table = table_ax.table(
            cellText=table_data,
            colLabels=cols,
            loc='center',
            cellLoc='center',
            colWidths=[0.1, 0.3, 0.3, 0.3]
            )
   
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)
   
        table_fig.suptitle('Peak Frequencies by Chunk (Lorentz fitted)', fontsize=12)
        table_fig.tight_layout()
        
        # Finalise the combined plot
        combined_ax.set_xlabel('Frequency (Hz)')
        combined_ax.set_ylabel('PSD ($V^2/Hz$)')
        combined_ax.set_title('Power Spectral Density for Different Time Chunks')
        combined_ax.grid(True, which='major', color='k', linestyle='-')
        combined_ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
        combined_ax.legend(loc='upper right', fontsize='small')
        
        
        # Create waterfall plot if requested
        if plot_waterfall and all_freqs and all_psds:
            # Create a meshgrid for 3D plotting
            all_freqs_np = _np.array(all_freqs)
            all_psds_np = _np.array(all_psds)
        
            # Ensure all frequency arrays are the same length for the meshgrid
            if all(len(freqs) == len(all_freqs[0]) for freqs in all_freqs):
                X, Y = _np.meshgrid(all_freqs[0], time_centers)
                Z = all_psds_np
            
                # Plot the waterfall diagram
                surf = waterfall_ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
                waterfall_ax.set_xlabel('Frequency (Hz)')
                waterfall_ax.set_ylabel('Time (s)')
                waterfall_ax.set_zlabel('PSD ($V^2$/Hz)')
                waterfall_ax.set_title('PSD Waterfall Plot')
            
                # Add a color bar
                waterfall_fig.colorbar(surf, ax=waterfall_ax, shrink=0.5, aspect=5)
            else:
                print("Warning: Frequency arrays have different lengths, cannot create waterfall plot")
        
        _plt.tight_layout()
        
        
        # Create peak amplitude drift plot
        amp_fig, amp_ax = _plt.subplots(figsize=(12, 8))

        # Plot model1 amplitudes
        if mode1_peaks:
            amp_ax.plot(mode1_times, mode1_amps, 'o-', color='red',
               label=f'Mode 1 amp: {_np.mean(mode1_amps):.2f}')
    
        # Plot model2 amplitudes
        if mode2_peaks:
            amp_ax.plot(mode2_times, mode2_amps, 'o-', color='magenta',
                   label=f'Mode 2 amp: {_np.mean(mode2_amps):.2f}')
    
        # Formatting
        amp_ax.set_xlabel('Time (s)')
        amp_ax.set_ylabel('Peak Amplitude ($V^2/Hz$)')
        amp_ax.set_title('Peak Amplitude Drift Over Time (Lorentzian Fitted)')
        amp_ax.grid(True)
        amp_ax.legend(loc='upper right', fontsize='small')
        _plt.tight_layout()
        _plt.show()
        
        # Return peak info
        return (mode1_times, mode1_freqs, mode1_amps, mode1_widths), (mode2_times, mode2_freqs, mode2_amps,mode2_widths), time_centers

def calc_PSD(Signal, SampleFreq, NPerSegment=1000000, window="hann"):
    freqs, PSD = scipy.signal.welch(Signal, SampleFreq,
                                    window=window, nperseg=NPerSegment)
    PSD = PSD[freqs.argsort()]
    freqs.sort()
    return freqs, PSD

def take_closest(myList, myNumber):
    pos = _bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def unit_conversion(array, unit_prefix, current_prefix=""):
    UnitDict = {
        'E': 1e18,
        'P': 1e15,
        'T': 1e12,
        'G': 1e9,
        'M': 1e6,
        'k': 1e3,
        '': 1,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
        'p': 1e-12,
        'f': 1e-15,
        'a': 1e-18,
    }
    try:
        Desired_units = UnitDict[unit_prefix]
    except KeyError:
        raise ValueError("You entered {} for the unit_prefix, this is not a valid prefix".format(unit_prefix))
    try:
        Current_units = UnitDict[current_prefix]
    except KeyError:
        raise ValueError("You entered {} for the current_prefix, this is not a valid prefix".format(current_prefix))
    conversion_multiplication = Current_units/Desired_units
    converted_array = array*conversion_multiplication
    return converted_array

def GenCmap(basecolor, ColorRange, NumOfColors, logscale=False):
    if NumOfColors > 256:
        _warnings.warn("Maximum Number of colors is 256", UserWarning)
        NumOfColors = 256
    if logscale == True:
        colors = [_sns.set_hls_values(basecolor, l=l) for l in _np.logspace(ColorRange[0], ColorRange[1], NumOfColors)]
    else:
        colors = [_sns.set_hls_values(basecolor, l=l) for l in _np.linspace(ColorRange[0], ColorRange[1], NumOfColors)]
    cmap = _sns.blend_palette(colors, as_cmap=True, n_colors=NumOfColors)
    return cmap

properties = {
    'default_fig_size': [6.5, 4],
    'default_linear_cmap': _sns.cubehelix_palette(n_colors=1024, light=1, as_cmap=True, rot=-.4),
    'default_log_cmap': GenCmap('green', [0, -60], 256, logscale=True),
    'default_base_color': 'green',
    }


def load_data(Filepath, ObjectType='data', RelativeChannelNo=None, SampleFreq=None, PointsToLoad=-1, calcPSD=True, NPerSegmentPSD=1000000, NormaliseByMonitorOutput=False, silent=False):
    if silent != True:
        print("Loading data from {}".format(Filepath))
    ObjectTypeDict = {
        'data' : DataObject,
        }
    try:
        Object = ObjectTypeDict[ObjectType]
    except KeyError:
        raise ValueError("You entered {}, this is not a valid object type".format(ObjectType))
    data = Object(filepath=Filepath, RelativeChannelNo=RelativeChannelNo, SampleFreq=SampleFreq, 
                  PointsToLoad=PointsToLoad, calcPSD=calcPSD, NPerSegmentPSD=NPerSegmentPSD, NormaliseByMonitorOutput=NormaliseByMonitorOutput)
    return data

def lorentzian(x, amplitude, center, width, offset):
    """
    Lorentzian/Cauchy distribution function.
    
    Parameters:
    -----------
    x : array
        The frequencies to evaluate the function at
    amplitude : float
        Peak amplitude
    center : float
        Center frequency (peak location)
    width : float
        Full width at half maximum (FWHM)
    offset : float
        Vertical offset
        
    Returns:
    --------
    array : Lorentzian function values
    """
    return offset + amplitude * (width**2 / ((x - center)**2 + width**2))

def fit_lorentzian_to_peaks(freqs, psd, mode_range, chunk_num=None):
    """
    Fit a Lorentzian function to a peak in the PSD within the specified mode range.
    Enhanced to handle noisier data and weaker peaks.
    
    Parameters:
    ----------
    freqs : array
        Frequency array
    psd : array
        Power spectral density array
    mode_range : tuple
        (min_freq, max_freq) range to search for the peak
    chunk_num : int, optional
        Chunk number for logging purposes
        
    Returns:
    --------
    tuple : (center, amplitude, width) or (None, None, None) if fit fails
    """
    # Find indices within the mode range
    mask = (freqs >= mode_range[0]) & (freqs <= mode_range[1])
    if not _np.any(mask):
        print(f"Chunk {chunk_num}: No data points in range {mode_range}")
        return None, None, None
    
    # Extract the data in this frequency range
    mode_freqs = freqs[mask]
    mode_psd = psd[mask]
    
    # Check for minimum points
    if len(mode_freqs) < 5:
        print(f"Chunk {chunk_num}: Not enough points ({len(mode_freqs)}) in range {mode_range}")
        return None, None, None
    
    # Identify potential peaks by comparing with local background
    # Calculate local noise level
    noise_level = _np.percentile(mode_psd, 25)  # Use 25th percentile as noise estimate
    
    # Apply smoothing to reduce impact of noise
    window_size = min(5, len(mode_psd) // 4)
    if window_size > 1:
        smooth_window = _np.hanning(window_size)
        smooth_window = smooth_window / _np.sum(smooth_window)
        # Use pandas rolling window for better handling of edges
        
        smooth_psd = pd.Series(mode_psd).rolling(window=window_size, center=True, min_periods=1).apply(
            lambda x: _np.sum(x * smooth_window[:len(x)]) / _np.sum(smooth_window[:len(x)])
        ).values
    else:
        smooth_psd = mode_psd

    # Find the maximum point in smoothed data
    peak_idx = _np.argmax(smooth_psd)
    peak_freq = mode_freqs[peak_idx]
    peak_amplitude = mode_psd[peak_idx]
    
    # Check if the peak is significantly above noise level
    snr = peak_amplitude / noise_level
    
    # Lower SNR threshold for weaker signals in later chunks
    min_snr = 1.5  # More lenient SNR threshold
    
    if snr < min_snr:
        print(f"Chunk {chunk_num}: Peak not significant enough (SNR={snr:.2f}) in range {mode_range}")
        
        # For weak peaks, try different approach - look for local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(smooth_psd, height=noise_level*min_snr, distance=len(smooth_psd)//4)
        
        if len(peaks) > 0:
            # Find highest peak amongst candidates
            best_peak = peaks[_np.argmax(smooth_psd[peaks])]
            peak_idx = best_peak
            peak_freq = mode_freqs[peak_idx]
            peak_amplitude = mode_psd[peak_idx]
            print(f"Chunk {chunk_num}: Found alternative peak at {peak_freq:.3f} Hz")
        else:
            # Still no peaks found, use more aggressive smoothing
            if len(mode_psd) > 10:
                extra_smooth = pd.Series(mode_psd).rolling(window=len(mode_psd)//3, center=True, min_periods=1).mean().values
                peak_idx = _np.argmax(extra_smooth)
                peak_freq = mode_freqs[peak_idx]
                peak_amplitude = mode_psd[peak_idx]
                print(f"Chunk {chunk_num}: Using heavily smoothed data to find peak at {peak_freq:.3f} Hz")
            else:
                return None, None, None
    
    # Estimate width from half-maximum points with improved robustness
    half_max = (peak_amplitude + noise_level) / 2
    
    # Find indices of points around half maximum
    above_half_max = smooth_psd >= half_max
    
    if _np.sum(above_half_max) >= 3:
        # Find the width based on half-maximum points
        indices = _np.where(above_half_max)[0]
        left_idx = indices[0]
        right_idx = indices[-1]
        
        # Calculate width, handling edge cases
        if right_idx - left_idx > 1:
            est_width = (mode_freqs[right_idx] - mode_freqs[left_idx]) / 2
        else:
            # Use a percentage of the frequency range if half-max points are too close
            est_width = (mode_range[1] - mode_range[0]) / 8
    else:
        # For very sharp or noisy peaks, use percentage of frequency range
        est_width = (mode_range[1] - mode_range[0]) / 8
    
    # Ensure width is reasonable (not too narrow or wide)
    min_width = (mode_range[1] - mode_range[0]) / 20
    max_width = (mode_range[1] - mode_range[0]) / 3
    est_width = _np.clip(est_width, min_width, max_width)
    
    # Initial parameter guess with vertical offset
    background = _np.percentile(mode_psd, 10)  # Estimate background as 10th percentile
    min_snr = 1.5
    
    # Calculate SNR for this peak
    peak_snr = peak_amplitude / noise_level

    # Adjust parameters for high amplitude peaks
    if peak_snr > 10:  # For very high SNR peaks
        # Use more conservative width estimate for high amplitude peaks
        est_width_high_amp = min(est_width, (mode_range[1] - mode_range[0]) / 12)
        
        # Adjust bounds to be more conservative for high amplitude peaks
        bounds = (
        # Lower bounds - tighter for high amplitude
        [peak_amplitude*0.7, mode_range[0] + (mode_range[1]-mode_range[0])*0.1, est_width_high_amp*0.5, 0],
        # Upper bounds - more constrained for high amplitude
        [peak_amplitude*1.5, mode_range[1] - (mode_range[1]-mode_range[0])*0.1, est_width_high_amp*2, peak_amplitude*0.1]
        )
    
        print(f"Chunk {chunk_num}: Using high-amplitude specific fitting parameters")
        # Use the adjusted width in initial guess
        p0 = [peak_amplitude - background, peak_freq, est_width_high_amp, background]
    else:  
        p0 = [peak_amplitude - background, peak_freq, est_width, background]
        # Set bounds for the parameters - wider and more flexible
        bounds = (
            [0, mode_range[0], min_width, 0],  # Lower bounds
            [peak_amplitude*3, mode_range[1], max_width, peak_amplitude]  # Upper bounds
            )
    
    # Define a more robust fitting function that handles outliers better
    def robust_fit(x, y, func, p0, bounds):
        try:
            # Try standard least squares first
            params, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, 
                                     method='trf', maxfev=3000)
            
            # Check fit quality
            fitted_values = func(x, *params)
            residuals = y - fitted_values
            ss_total = _np.sum((y - _np.mean(y))**2)
            ss_residual = _np.sum(residuals**2)
            r_squared = 1 - (ss_residual / ss_total)
            # Define the region close to the peak for focused analysis
            center = params[1]  # Peak center frequency
            width = params[2]   # Fitted width
            
            # Find points within ±3*width of the center (adjust multiplier as needed)
            peak_region_mask = _np.abs(mode_freqs - center) < (3 * width)

        # If we have enough points in this region
            if _np.sum(peak_region_mask) >= 5:
                # Calculate peak-specific residuals and R²
                peak_residuals = residuals[peak_region_mask]
                peak_data = mode_psd[peak_region_mask]
                peak_ss_total = _np.sum((peak_data - _np.mean(peak_data))**2)
                peak_ss_residual = _np.sum(peak_residuals**2)
    
                # Avoid division by zero
                if peak_ss_total > 0:
                    peak_r_squared = 1 - (peak_ss_residual / peak_ss_total)
                    
                    # Log the peak-specific quality
                    print(f"Chunk {chunk_num}: Peak-specific fit quality (R²={peak_r_squared:.3f}) at {center:.3f} Hz")
                
                    # Decision making- Is the fit any good?
                    if peak_r_squared < 0.7 and r_squared > 0.3:
                        print(f"Chunk {chunk_num}: Warning - Good global fit but poor fit near peak")
        
                    # Alternatively, use a weighted average of global and peak R²
                    combined_r_squared = (r_squared + 2*peak_r_squared) / 3
                    print(f"Chunk {chunk_num}: Combined fit quality (R²={combined_r_squared:.3f})")
            if r_squared < 0.3:  # lenient threshold for weak signals
                # Try again with weights to reduce influence of outliers
                weights = 1.0 / (1.0 + _np.abs(residuals/_np.std(residuals)))
                params, pcov = curve_fit(func, x, y, p0=params, bounds=bounds,
                                         sigma=1.0/weights, absolute_sigma=True,
                                         method='trf', maxfev=3000)
                
                # Recalculate fit quality
                fitted_values = func(x, *params)
                r_squared = 1 - _np.sum((y - fitted_values)**2) / ss_total
            
            return params, r_squared
            
        except Exception as e:
            print(f"Chunk {chunk_num}: Fit failed - {str(e)}")
            return None, -1
    
    # Perform robust fitting
    params, r_squared = robust_fit(mode_freqs, mode_psd, lorentzian, p0, bounds)
    
    if params is None:
        return None, None, None
    
    amplitude, center, width, offset = params
    
    # Sanity check the fit results with flexible thresholds for later chunks
    if r_squared < 0.2:  # Very low threshold to catch weak but real peaks
        print(f"Chunk {chunk_num}: Poor fit quality (R²={r_squared:.2f}) at {center:.3f} Hz")
        # We'll still return the results for manual inspection
    
    # Check if center is within reasonable bounds
    if center < mode_range[0] or center > mode_range[1]:
        print(f"Chunk {chunk_num}: Fitted center {center:.3f} Hz outside range {mode_range}")
        return None, None, None
    
    return center, amplitude, width

def analyse_multiple_files(file_paths, chunk_size=250, freq_range=[98, 102], plot_individual=False):
    """
    Analyse multiple .trc files in sequence and combine results for long-term stability analysis.
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to process
    chunk_size : float
        Size of each chunk in seconds
    freq_range : list
        [min_freq, max_freq] to focus on specific frequency range
    plot_individual : bool
        Whether to generate plots for each individual file
        
    Returns:
    --------
    tuple
        Combined results for all processed files
    """
    # Initialize containers for combined results
    all_mode1_times = []
    all_mode1_freqs = []
    all_mode1_amps = []
    all_mode1_widths = []
    
    all_mode2_times = []
    all_mode2_freqs = []
    all_mode2_amps = []
    all_mode2_widths = []
    
    # Track time offset for continuous timeline
    time_offset = 0
    
    # Process each file
    for i, file_path in enumerate(file_paths):
        print(f"\nProcessing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load data
        try:
            data_obj = load_data(file_path)
            
            # Get the actual time range for this file
            if hasattr(data_obj.time, 'get_array'):
                time_array = data_obj.time.get_array()
            else:
                time_array = _np.array(list(data_obj.time))
                
            file_start = time_array[0] if len(time_array) > 0 else 0
            file_end = time_array[-1] if len(time_array) > 0 else 0
            file_duration = file_end - file_start
            
            print(f"File time range: {file_start} to {file_end} (duration: {file_duration}s)")
            
            # Analyze drift for this file
            if plot_individual:
                # If plotting individual results, use the built-in method
                (mode1_data, mode2_data, chunk_centers) = data_obj.analyse_drift(
                    chunk_size=chunk_size, 
                    freq_range=freq_range, 
                    plot_waterfall=False
                )
            else:
                # Suppress plotting by modifying the method call
                # Store the original plot function
                original_plt_show = _plt.show
                # Replace with dummy function
                _plt.show = lambda: None
                
                # Call analyse_drift
                (mode1_data, mode2_data, chunk_centers) = data_obj.analyse_drift(
                    chunk_size=chunk_size, 
                    freq_range=freq_range, 
                    plot_waterfall=False
                )
                
                # Restore original plot function
                _plt.show = original_plt_show
            
            # Unpack results
            mode1_times, mode1_freqs, mode1_amps, mode1_widths = mode1_data
            mode2_times, mode2_freqs, mode2_amps, mode2_widths = mode2_data
            
            # For debugging
            if mode1_times:
                print(f"File {i+1} mode1 time range: {min(mode1_times)} to {max(mode1_times)}")
            
            # Adjust times by offset for continuous timeline
            if mode1_times:
                adjusted_mode1_times = [t + time_offset for t in mode1_times]
                all_mode1_times.extend(adjusted_mode1_times)
                all_mode1_freqs.extend(mode1_freqs)
                all_mode1_amps.extend(mode1_amps)
                all_mode1_widths.extend(mode1_widths)
                
            if mode2_times:
                adjusted_mode2_times = [t + time_offset for t in mode2_times]
                all_mode2_times.extend(adjusted_mode2_times)
                all_mode2_freqs.extend(mode2_freqs)
                all_mode2_amps.extend(mode2_amps)
                all_mode2_widths.extend(mode2_widths)
                
            # Update time offset for next file - use the actual file duration
            # This ensures we account for the full 20,000 seconds per file
            time_offset += file_duration
            
            print(f"Completed processing file: {os.path.basename(file_path)}")
            print(f"New cumulative time offset: {time_offset}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create the combined stability plots
    create_combined_stability_plots(
        all_mode1_times, all_mode1_freqs, all_mode1_amps,
        all_mode2_times, all_mode2_freqs, all_mode2_amps,
        file_paths,
    )
    
    return (all_mode1_times, all_mode1_freqs, all_mode1_amps), (all_mode2_times, all_mode2_freqs, all_mode2_amps)

def create_combined_stability_plots(mode1_times, mode1_freqs, mode1_amps, 
                                   mode2_times, mode2_freqs, mode2_amps,
                                   file_paths, mode1_widths=None, mode2_widths=None):
    """
    Create plots showing the combined stability across all processed files with error bars.
    
    Parameters:
    -----------
    mode1_times, mode1_freqs, mode1_amps : lists
        Data for mode 1 peaks
    mode2_times, mode2_freqs, mode2_amps : lists
        Data for mode 2 peaks
    file_paths : list
        List of processed file paths
    mode1_widths, mode2_widths : lists, optional
        Peak widths from Lorentzian fits, used for error bars
    """
    # Create markers to indicate file transitions
    file_transition_times = []
    current_offset = 0
    
    # We only need this for annotation, approximate is fine
    for i in range(len(file_paths)-1):
        # Estimate transition time as midpoint between last point of current file
        # and first point of next file
        if i < len(file_paths)-1:
            # Find last time in current file's data and first time in next file's data
            if i < len(file_paths)-1:
                current_file_times = [t for t in mode1_times + mode2_times 
                                     if t <= current_offset + 3000]  # Approximate file duration
                if current_file_times:
                    last_time = max(current_file_times)
                    current_offset = last_time
                    file_transition_times.append(last_time)
    
    # 1. Create frequency drift plot
    _plt.figure(figsize=(15, 8))
    
    # Plot mode 1 frequencies
    if mode1_freqs:
        # If we have peak widths available, use them for error bars
        if mode1_widths:
            # Calculate error estimate from peak widths
            # FWHM/SNR is a common estimate for frequency uncertainty
            # If you don't have SNR values, we can estimate based on amplitudes
            mode1_snr = _np.array(mode1_amps) / _np.median(_np.array(mode1_amps)/10)  # Rough SNR estimate
            mode1_errors = _np.array(mode1_widths) / _np.sqrt(mode1_snr)
            
            # Plot with error bars
            _plt.errorbar(mode1_times, mode1_freqs, yerr=mode1_errors, fmt='o-', 
                        color='blue', label=f"Mode 1: {_np.mean(mode1_freqs):.2f} Hz", 
                        alpha=0.7, capsize=3)
        else:
            # If no widths provided, estimate error from scatter
            # Use the standard deviation of adjacent points as error estimate
            mode1_errors = []
            for i in range(1, len(mode1_freqs)-1):
                local_std = _np.std([mode1_freqs[i-1], mode1_freqs[i], mode1_freqs[i+1]])
                mode1_errors.append(local_std)
            
            # Add first and last point errors (use nearest neighbor)
            if len(mode1_freqs) > 1:
                mode1_errors.insert(0, abs(mode1_freqs[1] - mode1_freqs[0]))
                mode1_errors.append(abs(mode1_freqs[-1] - mode1_freqs[-2]))
            else:
                # If only one point, use a small default error
                mode1_errors = [0.001]
            
            # Plot with error bars
            _plt.errorbar(mode1_times, mode1_freqs, yerr=mode1_errors, fmt='o-', 
                        color='blue', label=f"Mode 1: {_np.mean(mode1_freqs):.2f} Hz", 
                        alpha=0.7, capsize=3)
        # Calculate statistics
        mode1_mean = _np.mean(mode1_freqs)
        mode1_std = _np.std(mode1_freqs)
        mode1_drift = max(mode1_freqs) - min(mode1_freqs)
        print(f"Mode 1 - Mean: {mode1_mean:.3f} Hz, Std: {mode1_std:.5f} Hz, Range: {mode1_drift:.5f} Hz")
        
        # Add trend line
        z = _np.polyfit(mode1_times, mode1_freqs, 1)
        p = _np.poly1d(z)
        _plt.plot(mode1_times, p(mode1_times), '--', color='darkblue', 
                 label=f"Mode 1 trend: {z[0]*3600:.2e} Hz/hour")
        
        # Add horizontal line at mean
        _plt.axhline(y=mode1_mean, color='lightblue', linestyle='-', alpha=0.3)
    

    
    _plt.xlabel('Time (s)')
    _plt.ylabel('Frequency (Hz)')
    _plt.title('Long-term Peak Frequency Stability Analysis')
    _plt.grid(True)
    _plt.legend(loc='upper right')
    
    
    # 2. Create detailed statistics table
    fig_stats = _plt.figure(figsize=(12, 4))
    ax_stats = fig_stats.add_subplot(111)
    ax_stats.axis('off')
    
    # Prepare statistics data
    if mode1_freqs and mode2_freqs:
        table_data = [
            ['Parameter', 'Mode 1', 'Mode 2'],
            ['Mean Frequency (Hz)', f"{_np.mean(mode1_freqs):.5f}", f"{_np.mean(mode2_freqs):.5f}"],
            ['Std Dev Frequency (Hz)', f"{_np.std(mode1_freqs):.5f}", f"{_np.std(mode2_freqs):.5f}"],
            ['Min Frequency (Hz)', f"{min(mode1_freqs):.5f}", f"{min(mode2_freqs):.5f}"],
            ['Max Frequency (Hz)', f"{max(mode1_freqs):.5f}", f"{max(mode2_freqs):.5f}"],
            ['Drift Range (Hz)', f"{max(mode1_freqs)-min(mode1_freqs):.5f}", f"{max(mode2_freqs)-min(mode2_freqs):.5f}"],
            ['Relative Drift (ppm)', f"{1e6*(max(mode1_freqs)-min(mode1_freqs))/_np.mean(mode1_freqs):.2f}", 
             f"{1e6*(max(mode2_freqs)-min(mode2_freqs))/_np.mean(mode2_freqs):.2f}"],
            ['Mean Amplitude ($V^2/Hz$)', f"{_np.mean(mode1_amps):.2e}", f"{_np.mean(mode2_amps):.2e}"]
        ]
    elif mode1_freqs:
        table_data = [
            ['Parameter', 'Mode 1'],
            ['Mean Frequency (Hz)', f"{_np.mean(mode1_freqs):.5f}"],
            ['Std Dev Frequency (Hz)', f"{_np.std(mode1_freqs):.5f}"],
            ['Min Frequency (Hz)', f"{min(mode1_freqs):.5f}"],
            ['Max Frequency (Hz)', f"{max(mode1_freqs):.5f}"],
            ['Drift Range (Hz)', f"{max(mode1_freqs)-min(mode1_freqs):.5f}"],
            ['Relative Drift (ppm)', f"{1e6*(max(mode1_freqs)-min(mode1_freqs))/_np.mean(mode1_freqs):.2f}"],
            ['Mean Amplitude ($V^2/Hz$)', f"{_np.mean(mode1_amps):.2e}"]
        ]
    elif mode2_freqs:
        table_data = [
            ['Parameter', 'Mode 2'],
            ['Mean Frequency (Hz)', f"{_np.mean(mode2_freqs):.5f}"],
            ['Std Dev Frequency (Hz)', f"{_np.std(mode2_freqs):.5f}"],
            ['Min Frequency (Hz)', f"{min(mode2_freqs):.5f}"],
            ['Max Frequency (Hz)', f"{max(mode2_freqs):.5f}"],
            ['Drift Range (Hz)', f"{max(mode2_freqs)-min(mode2_freqs):.5f}"],
            ['Relative Drift (ppm)', f"{1e6*(max(mode2_freqs)-min(mode2_freqs))/_np.mean(mode2_freqs):.2f}"],
            ['Mean Amplitude ($V^2/Hz$)', f"{_np.mean(mode2_amps):.2e}"]
        ]
    
    # Create the table
    table = ax_stats.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.25, 0.25]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add headers
    for i, key in enumerate(table_data[0]):
        table[(0, i)].set_facecolor('#D7E4BC')
    
    _plt.title('Long-term Stability Statistics', fontsize=14)
    _plt.tight_layout()
    
    _plt.show()

# Main execution code
if __name__ == "__main__":
    # Base directory where files are located
    base_dir = "C:/Users/migsh/desktop/Zicheng's data for reference"
    
    # Find all trc files and sort them numerically
    file_pattern = os.path.join(base_dir, "C2--3Sepdata--*.trc")
    file_paths = sorted(glob.glob(file_pattern),
                       key=lambda x: int(x.split('--')[-1].split('.')[0]))
    
    print(f"Found {len(file_paths)} files to process")
    
    # Analyze all files sequentially
    results = analyse_multiple_files(
        file_paths,
        chunk_size=250,          # seconds per chunk
        freq_range=[11, 14],    # frequency range to analyze
        plot_individual=False    # don't show individual file plots
    )
    
    # Results now contain combined data across all files
    (all_mode1_times, all_mode1_freqs, all_mode1_amps), (all_mode2_times, all_mode2_freqs, all_mode2_amps) = results