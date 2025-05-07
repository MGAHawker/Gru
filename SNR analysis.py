# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 21:13:49 2025

@author: migsh
"""
import optoanalysis as opt
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def calc_PSD(Signal, SampleFreq, NPerSegment=1000000, window="hann"):
    """
    Extracts the pulse spectral density (PSD) from the data.

    Parameters
    ----------
    Signal : array-like
        Array containing the signal to have the PSD calculated for
    SampleFreq : float
        Sample frequency of the signal array
    NPerSegment : int, optional
        Length of each segment used in scipy.welch
        default = 1000000
    window : str or tuple or array_like, optional
        Desired window to use. See get_window for a list of windows
        and required parameters. If window is array_like it will be
        used directly as the window and its length will be used for
        nperseg.
        default = "hann"

    Returns
    -------
    freqs : ndarray
            Array containing the frequencies at which the PSD has been
            calculated
    PSD : ndarray
            Array containing the value of the PSD at the corresponding
            frequency value in V**2/Hz
    """
    freqs, PSD = scipy.signal.welch(Signal, SampleFreq,
                                    window=window, nperseg=NPerSegment)
    PSD = PSD[freqs.argsort()]
    freqs.sort()
    return freqs, PSD

def calculate_snr(freqs, psd, target_freq, bandwidth=1.0):
    """
    Calculate Signal-to-Noise Ratio for a given target frequency
    using pre-computed PSD data.
    
    Parameters:
    - freqs: Frequency array (output from calc_PSD)
    - psd: Power spectral density array (output from calc_PSD)
    - target_freq: Target frequency in Hz to analyze
    - bandwidth: Frequency bandwidth around target to consider as signal
    
    Returns:
    - snr: Signal-to-Noise Ratio in dB
    - signal_power: Calculated signal power
    - noise_power: Calculated noise power
    """
    # Find signal power in target frequency band
    signal_mask = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
    signal_power = np.sum(psd[signal_mask])
    
    # Noise power (excluding the signal band)
    noise_mask = ~signal_mask
    noise_power = np.mean(psd[noise_mask])
    
    # Calculate SNR
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return snr, signal_power, noise_power

def analyze_interferometer_data(pre_data, post_data, pre_freqs, post_freqs, bandwidth=1.0, nperseg=None):
    """
    Analyze and compare pre and post improvement interferometer data.
    
    Parameters:
    - pre_data: Tuple of (time_array, voltage_array) for pre-improvement data
    - post_data: Tuple of (time_array, voltage_array) for post-improvement data
    - pre_freqs: Dict of mode frequencies for pre-improvement data {'z_mode': freq, 'xy_mode': freq}
    - post_freqs: Dict of mode frequencies for post-improvement data {'z_mode': freq, 'xy_mode': freq}
    - bandwidth: Frequency bandwidth to consider around target frequency (Hz)
    - nperseg: Number of samples per segment for PSD calculation (default: auto-determined)
    
    Returns:
    - Dictionary with analysis results for Z and XY modes
    """
    # Unpack data
    pre_time, pre_voltage = pre_data
    post_time, post_voltage = post_data
    
    results = {}
    
    # Calculate sampling frequencies
    pre_sampling_freq = 1 / np.mean(np.diff(pre_time))
    post_sampling_freq = 1 / np.mean(np.diff(post_time))
    
    # Determine nperseg if not provided
    if nperseg is None:
        pre_nperseg = min(1000000, len(pre_voltage) // 4)  # Adjust as needed for your data
        post_nperseg = min(1000000, len(post_voltage) // 4)
    else:
        pre_nperseg = min(nperseg, len(pre_voltage))
        post_nperseg = min(nperseg, len(post_voltage))
    
    # Calculate PSDs using your function
    pre_freqs_psd, pre_psd = calc_PSD(pre_voltage, pre_sampling_freq, NPerSegment=pre_nperseg)
    post_freqs_psd, post_psd = calc_PSD(post_voltage, post_sampling_freq, NPerSegment=post_nperseg)
    
    # Create a figure for our plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time domain plots
    max_plot_samples = 10000  # Limit for clearer plots
    
    # Pre-improvement time domain
    plot_idx_pre = np.linspace(0, len(pre_time)-1, min(max_plot_samples, len(pre_time))).astype(int)
    axes[0, 0].plot(pre_time[plot_idx_pre], pre_voltage[plot_idx_pre])
    axes[0, 0].set_title('Pre-improvement Time Domain')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Voltage')
    
    # Post-improvement time domain
    plot_idx_post = np.linspace(0, len(post_time)-1, min(max_plot_samples, len(post_time))).astype(int)
    axes[0, 1].plot(post_time[plot_idx_post], post_voltage[plot_idx_post])
    axes[0, 1].set_title('Post-improvement Time Domain')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Voltage')
    
    # PSD plots
    axes[1, 0].semilogy(pre_freqs_psd, pre_psd)
    axes[1, 0].set_title('Pre-improvement PSD')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power/Frequency (V²/Hz)')
    
    axes[1, 1].semilogy(post_freqs_psd, post_psd)
    axes[1, 1].set_title('Post-improvement PSD')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power/Frequency (V²/Hz)')
    
    # Analysis for each mode
    modes = ['z_mode', 'xy_mode']
    colors = ['r', 'g']
    
    for i, mode in enumerate(modes):
        pre_target_freq = pre_freqs[mode]
        post_target_freq = post_freqs[mode]
        
        # Mark frequencies on plots
        axes[1, 0].axvline(x=pre_target_freq, color=colors[i], linestyle='--', 
                          label=f'{mode}: {pre_target_freq} Hz')
        axes[1, 1].axvline(x=post_target_freq, color=colors[i], linestyle='--',
                          label=f'{mode}: {post_target_freq} Hz')
        
        # Calculate SNR for both datasets
        pre_snr, pre_signal, pre_noise = calculate_snr(pre_freqs_psd, pre_psd, pre_target_freq, bandwidth)
        post_snr, post_signal, post_noise = calculate_snr(post_freqs_psd, post_psd, post_target_freq, bandwidth)
        
        # Calculate improvement
        snr_improvement = post_snr - pre_snr
        improvement_factor = 10 ** (snr_improvement / 10)
        
        # Calculate noise reduction ratio
        noise_reduction_ratio = pre_noise / post_noise if post_noise > 0 else float('inf')
        
        # Print results
        print(f"\n----- {mode.upper()} Analysis -----")
        print(f"Pre-improvement frequency: {pre_target_freq} Hz")
        print(f"Post-improvement frequency: {post_target_freq} Hz")
        print(f"Pre-improvement SNR: {pre_snr:.2f} dB")
        print(f"Post-improvement SNR: {post_snr:.2f} dB")
        print(f"SNR Improvement: {snr_improvement:.2f} dB")
        print(f"Improvement factor: {improvement_factor:.2f}x")
        print(f"Noise reduction ratio: {noise_reduction_ratio:.2f}x")
        
        # Store results
        results[mode] = {
            'pre_freq': pre_target_freq,
            'post_freq': post_target_freq,
            'pre_snr': pre_snr,
            'post_snr': post_snr,
            'pre_signal_power': pre_signal,
            'post_signal_power': post_signal,
            'pre_noise_power': pre_noise,
            'post_noise_power': post_noise,
            'improvement_db': snr_improvement,
            'improvement_factor': improvement_factor,
            'noise_reduction_ratio': noise_reduction_ratio
        }
    
    # Add legends to frequency plots
    axes[1, 0].legend()
    axes[1, 1].legend()
    
    # Optional: Add a zoomed view around the frequencies of interest
    # Create a new figure for zoomed PSD view
    fig_zoom, axes_zoom = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine zoom range - expand a bit beyond the bandwidth
    pre_min_freq = min(pre_freqs.values()) - bandwidth * 2
    pre_max_freq = max(pre_freqs.values()) + bandwidth * 2
    post_min_freq = min(post_freqs.values()) - bandwidth * 2
    post_max_freq = max(post_freqs.values()) + bandwidth * 2
    
    # Pre-improvement zoomed PSD
    axes_zoom[0].semilogy(pre_freqs_psd, pre_psd)
    axes_zoom[0].set_title('Pre-improvement PSD (Zoomed)')
    axes_zoom[0].set_xlabel('Frequency (Hz)')
    axes_zoom[0].set_ylabel('Power/Frequency (V²/Hz)')
    axes_zoom[0].set_xlim(pre_min_freq, pre_max_freq)
    
    # Post-improvement zoomed PSD
    axes_zoom[1].semilogy(post_freqs_psd, post_psd)
    axes_zoom[1].set_title('Post-improvement PSD (Zoomed)')
    axes_zoom[1].set_xlabel('Frequency (Hz)')
    axes_zoom[1].set_ylabel('Power/Frequency (V²/Hz)')
    axes_zoom[1].set_xlim(post_min_freq, post_max_freq)
    
    # Add mode markers to zoomed plots
    for i, mode in enumerate(modes):
        pre_target_freq = pre_freqs[mode]
        post_target_freq = post_freqs[mode]
        
        # Show the bandwidth regions used for SNR calculation
        axes_zoom[0].axvline(x=pre_target_freq, color=colors[i], linestyle='--', 
                           label=f'{mode}: {pre_target_freq} Hz')
        axes_zoom[0].axvspan(pre_target_freq - bandwidth/2, pre_target_freq + bandwidth/2, 
                           alpha=0.2, color=colors[i])
        
        axes_zoom[1].axvline(x=post_target_freq, color=colors[i], linestyle='--', 
                           label=f'{mode}: {post_target_freq} Hz')
        axes_zoom[1].axvspan(post_target_freq - bandwidth/2, post_target_freq + bandwidth/2, 
                           alpha=0.2, color=colors[i])
    
    axes_zoom[0].legend()
    axes_zoom[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

# How to use with actual data:
""
# Extracted data
pre = opt.load_data("C:/Users/migsh/Desktop/Zicheng's data for reference/C2--3Sepdata--00018.trc")
pre_time_array, pre_voltage_array = pre.get_time_data()

post = opt.load_data("C:/Users/migsh/Desktop/moon325/C2--Trace--00015.trc")
post_time_array, post_voltage_array = post.get_time_data()

# Define the mode frequencies for each dataset
pre_frequencies = {
    'z_mode': 13,  # Replace with your actual pre-improvement Z mode frequency
    'xy_mode': 12.5  # Replace with your actual pre-improvement XY mode frequency
}

post_frequencies = {
    'z_mode': 100.44,  # Your post-improvement Z mode frequency
    'xy_mode': 98.85   # Your post-improvement XY mode frequency
}

#Truncate and shift data to ensure equal durations
max_time = min(pre_time_array[-1] - pre_time_array[0], 
              post_time_array[-1] - post_time_array[0])

# Find indices to truncate
pre_idx = np.where(pre_time_array - pre_time_array[0] <= max_time)[0]
post_idx = np.where(post_time_array - post_time_array[0] <= max_time)[0]

# Truncate the data
pre_time_truncated = pre_time_array[pre_idx]
pre_voltage_truncated = pre_voltage_array[pre_idx]
post_time_truncated = post_time_array[post_idx]
post_voltage_truncated = post_voltage_array[post_idx]

# Shift time arrays to start at zero for both datasets
pre_time_shifted = pre_time_truncated - pre_time_truncated[0]
post_time_shifted = post_time_truncated - post_time_truncated[0]

# Pack the modified data
pre_data = (pre_time_shifted, pre_voltage_truncated)
post_data = (post_time_shifted, post_voltage_truncated)

# Run the analysis
results = analyze_interferometer_data(
    pre_data, 
    post_data, 
    pre_frequencies, 
    post_frequencies,
    bandwidth=0.5,    # Adjust based on how narrow your peaks are
    nperseg=1000000   # Use your preferred value from your existing PSD function
)