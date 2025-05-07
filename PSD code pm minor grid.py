# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:21:32 2025

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
        Analyze drift of peaks by dividing data into chunks and calculating PSD for each.
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
    
        # Process each chunk
        for chunk_idx in range(num_chunks):
            # Calculate time boundaries for this chunk
            chunk_start_time = start_time + chunk_idx * chunk_size
            chunk_end_time = min(chunk_start_time + chunk_size, end_time)
            chunk_center = (chunk_start_time + chunk_end_time) / 2
            time_centers.append(chunk_center)
            chunk_labels.append(f"t={chunk_start_time:.1f}s to {chunk_end_time:.1f}s")
        
            # Get indices for data in this chunk
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
        
            # Find peaks
            peak_indices = scipy.signal.find_peaks(plot_psd, height=_np.max(plot_psd)*0.1, distance=10)[0]
            chunk_peak_freqs = plot_freqs[peak_indices]
            chunk_peak_powers = plot_psd[peak_indices]
        
            # Sort peaks by power
            sorted_indices = _np.argsort(chunk_peak_powers)[::-1]
            top_peaks_freqs = chunk_peak_freqs[sorted_indices][:5]  # Keep top 5 peaks
            top_peaks_powers = chunk_peak_powers[sorted_indices][:5]
        
            peak_freqs.append(top_peaks_freqs)
            peak_powers.append(top_peaks_powers)
        
            # Plot PSD for this chunk on combined plot
            combined_ax.semilogy(plot_freqs, plot_psd, color=colours[chunk_idx], 
                            alpha=0.8, label=f"Chunk {chunk_idx+1}: {chunk_start_time:.1f}s-{chunk_end_time:.1f}s")
        
            # Mark peaks on the combined plot
            combined_ax.plot(top_peaks_freqs, top_peaks_powers, 'o', color=colours[chunk_idx])
        
            # Add text labels for top 2 peaks only to avoid clutter
            for i, (f, p) in enumerate(zip(top_peaks_freqs[:2], top_peaks_powers[:2])):
                combined_ax.text(f, p*1.1, f"{f:.2f} Hz", fontsize=8, color=colours[chunk_idx])
    
        # Finalize combined plot
        combined_ax.set_xlabel("Frequency (Hz)")
        combined_ax.set_ylabel("PSD ($V^2$/Hz)")
        combined_ax.set_title("Power Spectral Density for Different Time Chunks")
        combined_ax.grid(True, which='major', color='k', linestyle='-')
        combined_ax.grid(which='minor', color='k', linestyle=':', alpha=0.5)
   
        # Add legend with smaller font to avoid overcrowding
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
        
        # Create peak drift plot
        drift_fig, drift_ax = _plt.subplots(figsize=(12, 8))
    
        # Plot drift of each peak
        unique_peaks = {}
        tolerance = 1.0  # Hz - tolerance for grouping peaks
    
        # Identify unique peaks across all chunks
        for chunk_idx, (t, freqs) in enumerate(zip(time_centers, peak_freqs)):
            for freq in freqs:
                found_match = False
                for key in unique_peaks:
                    if abs(freq - key) < tolerance:
                        unique_peaks[key].append((chunk_idx, t, freq))
                        found_match = True
                        break
                if not found_match:
                    unique_peaks[freq] = [(chunk_idx, t, freq)]
    
        # Filter to keep only peaks that appear in at least 2 chunks
        persistent_peaks = {k: v for k, v in unique_peaks.items() if len(v) >= 2}
    
        # Plot drift for persistent peaks
        drift_colors = _plt.cm.tab10(_np.linspace(0, 1, len(persistent_peaks)))
        for i, (base_freq, points) in enumerate(persistent_peaks.items()):
            chunk_indices, times, freqs = zip(*points)
            drift_ax.plot(times, freqs, 'o-', color=drift_colors[i], label=f"{base_freq:.2f} Hz")
        
            # Calculate and annotate drift rate
            if len(times) > 1:
                drift_rate = (freqs[-1] - freqs[0]) / (times[-1] - times[0])
                drift_ax.text(times[-1], freqs[-1], f"{drift_rate:.3f} Hz/s", fontsize=8)
    
        drift_ax.set_xlabel('Time (s)')
        drift_ax.set_ylabel('Frequency (Hz)')
        drift_ax.set_title('Peak Frequency Drift Over Time')
        drift_ax.grid(True)
        drift_ax.legend()
    
        _plt.tight_layout()
        _plt.show()
    
        return peak_freqs, peak_powers, time_centers

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



#Now use said code

detector1 = load_data("C:/Users/migsh/Desktop/Zicheng's data for reference/C2--3SepData--00019.trc")

get_ipython().run_line_magic('matplotlib', 'qt')

fig, ax = detector1.plot_improved_PSD([0,130]);
