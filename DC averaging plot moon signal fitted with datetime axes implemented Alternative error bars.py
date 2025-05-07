# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:55:29 2025

@author: migsh
"""

import numpy as np
import matplotlib.pyplot as plt
import optoanalysis as opt
import glob
import os
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy.fft import fft, fftfreq
import matplotlib.ticker as ticker
import ephem 
from scipy import signal


# Get all the .trc files, manual loading would be ridiculously long
base_dir = "D:/"
file_pattern = os.path.join(base_dir, "C2--moon133--*.trc")
file_paths = sorted(glob.glob(file_pattern),
                   key=lambda x: int(x.split('--')[-1].split('.')[0]))

# Lists to store data from each file
data_segments = []
time_segments = []

# Load all the data segments
for file_path in file_paths:
    # Load data from files collected above
    data = opt.load_data(file_path)
    time_data, voltage = data.get_time_data()
    
    # Store voltage data as segments
    data_segments.append(voltage)
    
    # Store time data for gap analysis
    time_segments.append(time_data)

# Get file creation timestamps
file_datetimes = [(file_path, datetime.fromtimestamp(os.path.getmtime(file_path)))
                 for file_path in file_paths]

file_datetimes.sort(key=lambda x: x[1])
sorted_file_paths = [fd[0] for fd in file_datetimes]
file_creation_times = [fd[1].timestamp() for fd in file_datetimes]
absolute_start_times = [dt - file_creation_times[0] for dt in file_creation_times]

# Check file durations
file_durations = []
for time_data in time_segments:
    file_durations.append(time_data[-1] - time_data[0])

print("File durations:", [f"{d:.2f}" for d in file_durations])
print("Absolute start times:", [f"{t:.2f}" for t in absolute_start_times])

# All the trc files have a relative time running from 0 to 2500, we need absolute time.
print("Creating new ordered segments with proper timing...")

# Create lists to store properly ordered segments
ordered_data_segments = []
ordered_time_segments = []
gap_indices = []

# Sample rates and downsampling
downsampling_factor = 10  # Adjust as needed but will affect sampling freq
original_sample_rate = 1000
downsampled_sample_rate = original_sample_rate / downsampling_factor

reference_start_time = None
# Process each segment in order
for i in range(len(data_segments)):
    # Get current segment data
    current_data = data_segments[i]
    current_time = time_segments[i]
    
    # IMPORTANT: Create new time array based on absolute file start time
    # Ignore the original time data which is relative (0-2500) for all files
    sample_count = len(current_data)
    file_duration = file_durations[i]
    
    # Create time array from 0 to file_duration with correct number of points
    time_array = np.linspace(0, file_duration, sample_count)
    if i == 0:
        reference_start_time = absolute_start_times[0]  # Store reference once
        # Shift all segments relative to the first segment's start time
    absolute_time = time_array + (absolute_start_times[i] - reference_start_time)
    
    # Add segment to ordered lists
    ordered_data_segments.append(current_data)
    ordered_time_segments.append(absolute_time)
    
    print(f"Segment {i}: Starts at {absolute_start_times[i]:.2f}s, ends at {absolute_start_times[i] + file_duration:.2f}")
    
    # Add gap if this isn't the last segment
    if i < len(data_segments) - 1:
        # Calculate when this file ends and next starts
        current_end_time = absolute_start_times[i] + file_duration
        next_start_time = absolute_start_times[i+1]
        
        # Calculate gap duration
        gap_duration = next_start_time - current_end_time
        print(f"Gap between segment {i} and {i+1}: {gap_duration:.2f} seconds")
        
        # Only add gap if it's significant (positive)
        if gap_duration > 0.1:  # Small threshold to avoid floating point issues
            # Create gap segment with baseline voltage
            baseline_voltage = 1.1  # or calculate average from segments
            gap_samples = int(gap_duration * downsampled_sample_rate)
            
            if gap_samples > 0:
                # Create gap segment
                gap_segment = np.ones(gap_samples) * baseline_voltage
                
                # Create time array for gap
                gap_time = np.linspace(
                    current_end_time - reference_start_time,
                    next_start_time - reference_start_time,
                    gap_samples
                )
                
                # Add gap to ordered lists
                ordered_data_segments.append(gap_segment)
                ordered_time_segments.append(gap_time)
                
                # Add to gap indices
                gap_indices.append(len(ordered_data_segments) - 1)
                
                print(f"  Added gap with {gap_samples} points")

# Print some diagnostic information
print(f"Original data segments: {sum(len(seg) for seg in data_segments)} points")
print(f"Ordered data segments: {sum(len(seg) for seg in ordered_data_segments)} points")






# Downsampling for performance
print("Downsampling data for performance...")
ordered_data_segments_downsampled = []
ordered_time_segments_downsampled = []

# Averaging downsample approach instead of keeping every nth point
for i, (time_data, data) in enumerate(zip(ordered_time_segments, ordered_data_segments)):
    # Calculate how many points will be in the downsampled result
    new_length = int(np.ceil(len(data) / downsampling_factor))
    
    # Initialise arrays for averaged data and corresponding timestamps
    downsampled_data = np.zeros(new_length)
    downsampled_time = np.zeros(new_length)
    
    # Perform averaging for each bin
    for j in range(new_length):
        start_idx = j * downsampling_factor
        end_idx = min((j + 1) * downsampling_factor, len(data))
        
        # Average the voltage values in this bin
        if start_idx < end_idx:
            downsampled_data[j] = np.mean(data[start_idx:end_idx])
            # For time, use the average time point in this bin
            downsampled_time[j] = np.mean(time_data[start_idx:end_idx])
    
    ordered_data_segments_downsampled.append(downsampled_data)
    ordered_time_segments_downsampled.append(downsampled_time)

print(f"Downsampled data segments: {sum(len(seg) for seg in ordered_data_segments_downsampled)} points")

# Rebuild gap indices for downsampled data
downsampled_gap_indices = []
current_index = 0
for i in range(len(ordered_data_segments)):
    if i in gap_indices:
        downsampled_gap_indices.append(current_index)
    current_index += 1

#Dear God cull the data there's way too many points too handle otherwise
downsampling_factor = 10  # Adjust as needed but will effect sampling freq 
data_segments_downsampled = []
time_segments_downsampled = []

#Averaging downsample approach instead of keep nth point
for i, (time_data, data) in enumerate(zip(time_segments, data_segments)):
    # Calculate how many points will be in the downsampled result
    new_length = int(np.ceil(len(data) / downsampling_factor))
    
    # Initialise arrays for averaged data and corresponding timestamps
    downsampled_data = np.zeros(new_length)
    downsampled_time = np.zeros(new_length)
    
    # Perform averaging for each bin
    for j in range(new_length):
        start_idx = j * downsampling_factor
        end_idx = min((j + 1) * downsampling_factor, len(data))
        
        # Average the voltage values in this bin
        if start_idx < end_idx:
            downsampled_data[j] = np.mean(data[start_idx:end_idx])
            # For time, use the average time point in this bin
            downsampled_time[j] = np.mean(time_data[start_idx:end_idx])
    
    data_segments_downsampled.append(downsampled_data)
    time_segments_downsampled.append(downsampled_time)
    
def time_based_downsampling(time_segments, voltage_segments, gap_indices, bin_duration=1800):
    """
    Downsample data based on time bins, excluding gap-filled segments
    
    Parameters:
    time_segments (list): List of time segment arrays
    voltage_segments (list): List of voltage segment arrays
    gap_indices (list): Indices of gap-filled segments to exclude
    bin_duration (float): Duration of each bin in seconds (default 1800 = 30 minutes)
    
    Returns:
    tuple: downsampled_time, downsampled_voltage
    """
    # Combine non-gap segments
    valid_time = []
    valid_voltage = []
    
    for i, (time_data, voltage_data) in enumerate(zip(time_segments, voltage_segments)):
        # Skip gap-filled segments
        if i in gap_indices:
            continue
        
        valid_time.append(time_data)
        valid_voltage.append(voltage_data)
    
    # Concatenate valid segments
    combined_time = np.concatenate(valid_time)
    combined_voltage = np.concatenate(valid_voltage)
    
    # Find the start and end of the entire dataset
    start_time = combined_time[0]
    end_time = combined_time[-1]
    
    # Create time bins
    time_bins = np.arange(start_time, end_time + bin_duration, bin_duration)
    
    # Initialise arrays to store downsampled data
    downsampled_time = []
    downsampled_voltage = []
    
    # Downsample for each time bin
    for i in range(len(time_bins) - 1):
        # Find indices within the current time bin
        bin_mask = (combined_time >= time_bins[i]) & (combined_time < time_bins[i+1])
        
        # If bin's not empty
        if np.any(bin_mask):
            bin_data = combined_voltage[bin_mask]
            bin_timestamps = combined_time[bin_mask]
            
            # Calculate mean voltage and mean time for bin
            mean_voltage = np.mean(bin_data)
            mean_time = np.mean(bin_timestamps)
            
            downsampled_time.append(mean_time)
            downsampled_voltage.append(mean_voltage)
    
    return np.array(downsampled_time), np.array(downsampled_voltage)

# Plotting code
plt.figure(figsize=(15, 10))

# Perform time-based downsampling
downsampled_time, downsampled_voltage = time_based_downsampling(
    ordered_time_segments, 
    ordered_data_segments, 
    downsampled_gap_indices,  # Use your existing gap indices
    bin_duration=1800  # 30 minute averaging following Tim's suggestion
)

# Plot original data segments
plt.subplot(2, 1, 1)
plt.title("Original Data Segments")
for i, (time_data, voltage_data) in enumerate(zip(ordered_time_segments, ordered_data_segments)):
    plt.plot(time_data, voltage_data, color=plt.cm.tab10(i % 10), alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (V)")
plt.grid(True, alpha=0.3)

# Plot downsampled data
plt.subplot(2, 1, 2)
plt.title("Downsampled 'DC trace' (30-minute bins)")
plt.plot(downsampled_time, downsampled_voltage, 'ro-')
plt.xlabel("Time (seconds)")
plt.ylabel("Average Voltage (V)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Define lunar period
lunar_period = 12 * 3600 + 25 * 60  # 12h 25m in seconds half-lunar day


# Step 1: Create a smoothed baseline trend
# Use Savitzky-Golay filter for smoothing - better than simple moving average
# Window size needs to be odd and larger than lunar period to capture non-lunar trends
lunar_period_points = int(lunar_period / 1800)  # Convert lunar period to number of 30-min points
window_size = 2 * lunar_period_points + 1  # Make it more than 2x lunar period and odd

# Check if we have enough data points for this window size
if len(downsampled_voltage) > window_size:
    # Apply Savitzky-Golay filter to create baseline
    baseline = savgol_filter(
        downsampled_voltage, 
        window_length=window_size, 
        polyorder=3  # Quadratic polynomial
    )
else:
    # If not enough points, use a simpler polynomial fit
    z = np.polyfit(downsampled_time, downsampled_voltage, 3)
    baseline = np.polyval(z, downsampled_time)

# Step 2: Subtract baseline to isolate potential lunar oscillations
lunar_component_base = downsampled_voltage - baseline

#Substep 2: try and filter out the dominant period
period_to_isolate = 11.44  # hours
frequency_to_isolate = 1 / (period_to_isolate * 3600)  # Hz

# Calculate sampling frequency
sampling_interval = np.mean(np.diff(downsampled_time))  # seconds
fs = 1 / sampling_interval  # Hz

print(f"Sampling frequency: {fs:.6f} Hz")
print(f"Target frequency to isolate: {frequency_to_isolate:.8f} Hz")

# Design a bandpass filter to isolate the 11.44h component
# Define bandwidth - how wide around the target frequency
bandwidth_percent = 2.5  # % width around the center frequency (Now redundant as I directly specify bandwidth)
bandwidth = 0.00000202

# Calculate filter edges
lowcut = frequency_to_isolate - (bandwidth/2)
highcut = frequency_to_isolate + (bandwidth/2)

# Ensure we don't go below 0 Hz
lowcut = max(lowcut, 1e-10)

# Create the bandpass filter
nyquist = fs / 2
low = lowcut / nyquist
high = highcut / nyquist
order = 4  # Filter order - affects steepness of rolloff

b, a = signal.butter(order, [low, high], btype='band')

# Apply the filter to isolate the 11.44h component
lunar_component = signal.filtfilt(b, a, lunar_component_base)

# Step 3: Fit the lunar model to these isolated oscillations
# Lunar model function
def lunar_model(t, amplitude, phase):
    lunar_freq = 2 * np.pi / lunar_period  # 12h 25m in seconds
    return amplitude * np.sin(lunar_freq * t + phase)

# Fit the model
try:
    # Initial guess: small amplitude and zero phase
    initial_guess = [0.005, 0]
    
    params, params_covariance = curve_fit(
        lunar_model,
        downsampled_time, 
        lunar_component_base,
        p0=initial_guess,
        maxfev=5000
    )
    
    # Generate the fitted lunar model
    fitted_lunar = lunar_model(downsampled_time, *params)
    
    # Calculate fit quality (R-squared)
    ss_tot = np.sum((lunar_component_base - np.mean(lunar_component_base))**2)
    ss_res = np.sum((lunar_component_base - fitted_lunar)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Lunar model fit parameters:")
    print(f"  Amplitude: {params[0]:.6f} V")
    print(f"  Phase: {params[1]:.4f} rad")
    print(f"  R-squared: {r_squared:.4f}")
    
except Exception as e:
    print(f"Error fitting lunar model: {e}")
    fitted_lunar = np.zeros_like(downsampled_time)
    r_squared = 0
    
def seconds_to_datetime(seconds_array, reference_datetime):
    """Convert array of seconds to datetime objects"""
    return [reference_datetime + timedelta(seconds=s) for s in seconds_array]
reference_datetime = datetime.fromtimestamp(file_creation_times[0])

# Step 4: Visualise the results
# Four-plot figure with datetime axis
plt.figure(figsize=(15, 10))

# Convert downsampled_time to datetime for plotting
datetime_data = seconds_to_datetime(downsampled_time, reference_datetime)

# Plot 1: Original data
plt.title("Original Downsampled Data")
plt.plot(datetime_data, downsampled_voltage, 'ro-')
plt.xlabel("Date and Time")
plt.ylabel("Average Voltage (V)")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=3))  # minor ticks every 3 hours
plt.gca().grid(True, which='minor', linestyle=':', alpha=0.4)  # Show minor grid lines

# Plot 2: Original data with baseline trend
plt.figure(figsize=(15, 10))
plt.title("Original Data with Baseline Trend")
plt.plot(datetime_data, downsampled_voltage, 'ro-', label="Original Data")
plt.plot(datetime_data, baseline, 'k-', linewidth=2, label="Baseline Trend")
plt.xlabel("Date and Time")
plt.ylabel("Average Voltage (V)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=3))  # minor ticks every 3 hours
plt.gca().grid(True, which='minor', linestyle=':', alpha=0.4)  # Show minor grid lines

# Plot 3: Isolated lunar component
plt.figure(figsize=(15, 10))
plt.title("Isolated Lunar Component (Data - Baseline)")
plt.plot(datetime_data, lunar_component_base, 'bo-')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel("Date and Time")
plt.ylabel("Lunar Component (V)")
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=3))  # minor ticks every 3 hours
plt.gca().grid(True, which='minor', linestyle=':', alpha=0.4)  # Show minor grid lines

# Plot 4: Isolated lunar component with fitted lunar model
plt.figure(figsize=(15, 10))
plt.title(f"Lunar Component with Fitted Model (R² = {r_squared:.4f})")
plt.plot(datetime_data, lunar_component_base, 'bo-', label="Lunar Component")
plt.plot(datetime_data, fitted_lunar, 'g-', linewidth=2, label="Lunar Model")
plt.xlabel("Date and Time")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=3))  # minor ticks every 3 hours
plt.gca().grid(True, which='minor', linestyle=':', alpha=0.4)  # Show minor grid lines

plt.tight_layout()
plt.show()

# Function to add error bars to FFT analysis using Welch's method
def fft_with_improved_period_errors(signal, sampling_time, total_duration):
    """
    Calculate FFT with more conservative period error estimates
    
    Parameters:
    signal: time-domain signal
    sampling_time: time between samples (seconds)
    total_duration: total duration of signal (seconds)
    
    Returns:
    periods, magnitudes, period_errors, amplitude_errors
    """
    N = len(signal)
    
    # Calculate FFT
    yf = fft(signal)
    xf = fftfreq(N, sampling_time)[:N//2]
    magnitudes = 2.0/N * np.abs(yf[:N//2])
    
    # Convert to periods (in seconds)
    with np.errstate(divide='ignore'):  # Ignore division by zero
        periods = 1.0 / xf[1:]  # Skip first element (DC component)
    
    # Also skip DC component for other arrays
    xf = xf[1:]
    magnitudes = magnitudes[1:]
    
    # Basic frequency resolution from signal length
    df = 1.0 / total_duration  # Hz
    
    # Improved period error calculation - more conservative approach
    # Error in period = period * relative error in frequency
    # Relative error in frequency is approximately df/f for each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_freq_error = df / xf
        period_errors = periods * relative_freq_error
    
    # Handle any NaN values
    period_errors = np.nan_to_num(period_errors)
    
    # Calculate bin spacing in period domain
    bin_spacings = np.diff(periods)
    min_period_errors = np.zeros_like(periods)
    min_period_errors[:-1] = bin_spacings / 2  # Half the bin width
    min_period_errors[-1] = min_period_errors[-2] if len(min_period_errors) > 1 else df  # Handle last point
    
    # Use maximum of calculated error and minimum error
    period_errors = np.maximum(period_errors, min_period_errors)
    
    # Scale factor to make error bars more realistic
    # This is empirical and can be adjusted based on your confidence level
    error_scale_factor = 0.5  # Reduce error bars by 50%
    period_errors *= error_scale_factor
    
    # Estimate amplitude errors using noise level
    noise_level = np.median(magnitudes[-int(N/10):]) if N > 20 else np.std(magnitudes)
    amplitude_errors = np.ones_like(magnitudes) * noise_level
    
    return periods, magnitudes, period_errors, amplitude_errors

# Apply to lunar component
T = np.mean(np.diff(downsampled_time))  # Average sampling interval
total_duration = downsampled_time[-1] - downsampled_time[0]  # Total signal duration

periods, mag, period_err, amp_err = fft_with_improved_period_errors(lunar_component, T, total_duration)

# Convert to hours for plotting
periods_hours = periods / 3600
period_errors_hours = period_err / 3600

# Create plot with error bars on both axes and hourly grid lines
plt.figure(figsize=(12, 6))
plt.title("FFT Analysis for Lunar Component with Period Uncertainty")

# Plot with error bars on both axes
plt.errorbar(periods_hours, mag, 
             xerr=period_errors_hours, yerr=amp_err, 
             fmt='o-', capsize=3, alpha=0.7)

# Add exact lunar period line
plt.axvline(x=12.42, color='r', linestyle='--', label="Expected Lunar Period (~12.42h)")

# Add day/night reference (half of 24 hours)
plt.axvline(x=12.0, color='g', linestyle='-.', alpha=0.7, label="Day/Night Cycle (12h)")

plt.xlim(0, 24)  # Focus on periods up to 24 hours
plt.ylim(bottom=0)  # Start y-axis from 0
plt.xlabel("Period (hours)")
plt.ylabel("Amplitude (V)")

# Add major and minor grid lines
plt.grid(True, which='major', linestyle='-', alpha=0.6)
plt.grid(True, which='minor', linestyle=':', alpha=0.4)

# Set minor ticks at every hour
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(4))  # Major tick every 4 hours
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # Minor tick every 1 hour

# Add vertical gridlines at hour intervals
for hour in range(1, 25):
    if hour % 4 != 0:  # Skip hours that already have major gridlines
        plt.axvline(x=hour, color='gray', linestyle=':', alpha=0.3)

# Annotate key periods
def find_nearest_idx(array, value):
    return np.argmin(np.abs(array - value))

# Find peaks near critical values
lunar_idx = find_nearest_idx(periods_hours, 12.42)
diurnal_idx = find_nearest_idx(periods_hours, 12.0)
eleven_hr_idx = find_nearest_idx(periods_hours, 11.4)

# Annotate with values
plt.annotate(f"{periods_hours[lunar_idx]:.2f}h", 
            xy=(periods_hours[lunar_idx], mag[lunar_idx]), 
            xytext=(5, 10), textcoords='offset points')

plt.annotate(f"{periods_hours[diurnal_idx]:.2f}h", 
            xy=(periods_hours[diurnal_idx], mag[diurnal_idx]), 
            xytext=(5, -15), textcoords='offset points')

if abs(periods_hours[eleven_hr_idx] - 11.4) < 1.0:  # Only if we have a point near 11.4
    plt.annotate(f"{periods_hours[eleven_hr_idx]:.2f}h", 
                xy=(periods_hours[eleven_hr_idx], mag[eleven_hr_idx]), 
                xytext=(5, 10), textcoords='offset points')

plt.legend()
plt.tight_layout()

# Optional: Apply same analysis to baseline with improved error bars
periods_base, mag_base, period_err_base, amp_err_base = fft_with_improved_period_errors(
   baseline, T, total_duration)

# Convert to hours for plotting
periods_base_hours = periods_base / 3600
period_errors_base_hours = period_err_base / 3600

# Create plot with error bars on both axes for baseline
plt.figure(figsize=(12, 6))
plt.title("FFT Analysis for Baseline with Period Uncertainty")

# Plot with error bars on both axes
plt.errorbar(periods_base_hours, mag_base, 
             xerr=period_errors_base_hours, yerr=amp_err_base, 
             fmt='o-', capsize=3, alpha=0.7)

# Add reference lines
plt.axvline(x=12.42, color='r', linestyle='--', label="Lunar Period (12.42h)")
plt.axvline(x=24.0, color='g', linestyle='-.', alpha=0.7, label="Solar Day (24h)")
plt.axvline(x=12.0, color='g', linestyle=':', alpha=0.7, label="Half-Day (12h)")

plt.xlim(0, 150)  # Focus on periods up to 100 hours
plt.ylim(bottom=0)  # Start y-axis from 0
plt.xlabel("Period (hours)")
plt.ylabel("Amplitude (V)")

#Add major and minor grid lines for baseline plot
plt.grid(True, which='major', linestyle='-', alpha=0.6)
plt.grid(True, which='minor', linestyle=':', alpha=0.4)

#Set minor ticks at appropriate intervals for the larger range
ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))  # Major tick every 10 hours
ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))   # Minor tick every 2 hours

#Add vertical gridlines at appropriate hour intervals
for hour in range(0, 101, 2):
    if hour % 10 != 0:  # Skip hours that already have major gridlines
        plt.axvline(x=hour, color='gray', linestyle=':', alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()



# ----- Moon Phase Analysis with ephem -----
print("Performing lunar phase analysis with ephem...")


# Function to calculate moon position at a given datetime
def get_moon_position(dt, observer_location):
    """
    Calculate moon altitude and azimuth for a specific datetime and location
    
    Parameters:
    dt (datetime): Date and time to calculate moon position
    observer_location (ephem.Observer): Observer location object
    
    Returns:
    tuple: (altitude in radians, azimuth in radians)
    """
    observer_location.date = ephem.Date(dt)
    moon = ephem.Moon(observer_location)
    return float(moon.alt), float(moon.az)

# Function to calculate gravitational pull from moon
def moon_gravity_model(dt, observer_location):
    """
    Calculate relative gravitational influence from the moon
    
    This is a simplified model proportional to 1/r² and sin(alt)
    where r is the Earth-Moon distance and alt is the moon's altitude
    
    Parameters:
    dt (datetime): Date and time to calculate
    observer_location (ephem.Observer): Observer location
    
    Returns:
    float: Relative gravitational effect (arbitrary units)
    """
    observer_location.date = ephem.Date(dt)
    moon = ephem.Moon(observer_location)
    
    # Get moon distance (in Earth radii)
    distance = moon.earth_distance
    
    # Get moon altitude (in radians)
    altitude = float(moon.alt)
    
    # Calculate vertical component of gravitational pull
    # Proportional to sin(altitude) and 1/distance²
    if altitude > 0:  # Moon is above horizon
        gravity_effect = np.sin(altitude) / (distance**2)
    else:  # Moon is below horizon
        # Still has some effect but diminished
        gravity_effect = np.sin(altitude) / (distance**2) * 1
        
    return gravity_effect

# Function for combined moon+sun model
def weighted_combined_model(dt, observer_location, moon_weight=1.0, sun_weight=0.5):
    """Calculate weighted combination of moon and sun effects"""
    observer_location.date = ephem.Date(dt)
    
    # Moon calculations
    moon = ephem.Moon(observer_location)
    moon_alt = float(moon.alt)
    moon_dist = moon.earth_distance
    moon_phase = float(moon.phase) / 100.0  # Moon phase (0-1)
    
    # Include both altitude effect and phase effect
    if moon_alt > 0:
        altitude_effect = np.sin(moon_alt)
    else:
        altitude_effect = np.sin(moon_alt) * 0.5  # Reduced effect when below horizon
    
    # Distance effect (inverse square law)
    distance_effect = 1.0 / (moon_dist**2)
    
    # Combine effects (altitude and distance are physical, phase is visual)
    moon_effect = altitude_effect * distance_effect
    
    # Sun calculations (similar approach)
    sun = ephem.Sun(observer_location)
    sun_alt = float(sun.alt)
    sun_dist = sun.earth_distance
    
    if sun_alt > 0:
        sun_altitude_effect = np.sin(sun_alt)
    else:
        sun_altitude_effect = np.sin(sun_alt) * 0.5
    
    sun_distance_effect = 1.0 / (sun_dist**2)
    sun_effect = sun_altitude_effect * sun_distance_effect * 0.5  # Scale sun effect
    
    # Weighted combined effect
    combined_effect = moon_weight * moon_effect + sun_weight * sun_effect
    
    return moon_effect, sun_effect, combined_effect, moon_phase

# Calculate cross-correlation to find phase relationship
def calculate_cross_correlation(signal1, signal2):
    """Calculate normalised cross-correlation between two signals"""
    # Ensure signals are normalised to zero mean
    s1 = signal1 - np.mean(signal1)
    s2 = signal2 - np.mean(signal2)
    
    # Calculate cross-correlation
    correlation = np.correlate(s1, s2, mode='full')
    
    # Normalize
    n = len(s1)
    correlation = correlation / (n * np.std(s1) * np.std(s2))
    
    # Create lag array
    lags = np.arange(-n+1, n)
    
    return lags, correlation

# ----- Main Analysis Code -----

# Set up observer location for the experiment
observer = ephem.Observer()
observer.lat = '50.93518'  # approx latitude
observer.lon = '-1.39956'  # approx longitude
observer.elevation = 41.5   # elevation in metres

# Use actual datetime data
reference_datetime = datetime.fromtimestamp(file_creation_times[0])
datetime_data = seconds_to_datetime(downsampled_time, reference_datetime)

print("Calculating celestial positions and gravitational effects...")
moon_alt = []
moon_az = []
moon_grav = []
sun_grav = []
combined_grav = []
moon_phases = []

for dt in datetime_data:
    alt, az = get_moon_position(dt, observer)
    moon_alt.append(alt)
    moon_az.append(az)
    
    # Calculate gravitational effects with improved model
    m_effect, s_effect, c_effect, m_phase = weighted_combined_model(dt, observer)
    moon_grav.append(m_effect)
    sun_grav.append(s_effect)
    combined_grav.append(c_effect)
    moon_phases.append(m_phase)

# Keep the raw values without normalization for debugging
raw_moon_grav = np.array(moon_grav)
raw_sun_grav = np.array(sun_grav)
moon_phases = np.array(moon_phases)

# Now normalise for comparison with lunar_component
moon_grav = np.array(moon_grav)
moon_grav_norm = (moon_grav - np.mean(moon_grav)) / np.std(moon_grav) * np.std(lunar_component) + np.mean(lunar_component)

sun_grav = np.array(sun_grav)
sun_grav_norm = (sun_grav - np.mean(sun_grav)) / np.std(sun_grav) * np.std(lunar_component) + np.mean(lunar_component)

# Calculate time in hours from start for correlation analysis
time_hours = [(dt - reference_datetime).total_seconds()/3600 for dt in datetime_data]

# Calculate cross-correlation
print("Calculating cross-correlations...")
lags_moon, corr_moon = calculate_cross_correlation(lunar_component, moon_grav)
lags_sun, corr_sun = calculate_cross_correlation(lunar_component, sun_grav)


# Calculate average time between samples (in hours)
avg_hours_per_sample = np.mean(np.diff(time_hours))
print(f"Average time between samples: {avg_hours_per_sample:.4f} hours")

# Convert lags to hours
lags_hours_moon = lags_moon * avg_hours_per_sample
lags_hours_sun = lags_sun * avg_hours_per_sample

# Find maximum correlation and corresponding lag
max_corr_moon_idx = np.argmax(np.abs(corr_moon))
max_corr_moon = corr_moon[max_corr_moon_idx]
max_lag_moon = lags_hours_moon[max_corr_moon_idx]

max_corr_sun_idx = np.argmax(np.abs(corr_sun))
max_corr_sun = corr_sun[max_corr_sun_idx]
max_lag_sun = lags_hours_sun[max_corr_sun_idx]



print(f"Moon correlation: {max_corr_moon:.4f} at lag {max_lag_moon:.2f} hours")
print(f"Sun correlation: {max_corr_sun:.4f} at lag {max_lag_sun:.2f} hours")


# --- Visualisation ---

# Plot 1: Time series comparison
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.title("Lunar Component vs Moon/Sun Gravitational Model")
plt.plot(datetime_data, lunar_component, 'bo-', label="Lunar Component", alpha=0.7)
plt.plot(datetime_data, moon_grav, 'r-', label="Moon Gravity Model", alpha=0.7)
plt.plot(datetime_data, sun_grav, 'g-', label="Sun Gravity Model", alpha=0.5)
#plt.plot(datetime_data, combined_grav, 'k--', label="Combined Model", alpha=0.7)
plt.xlabel("Date and Time")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Plot 2: Cross-correlation
plt.subplot(2, 1, 2)
plt.title("Cross-Correlation Analysis")
plt.plot(lags_hours_moon, corr_moon, 'r-', label=f"Moon (Max: {max_corr_moon:.3f} at {max_lag_moon:.2f}h)")
plt.plot(lags_hours_sun, corr_sun, 'g-', label=f"Sun (Max: {max_corr_sun:.3f} at {max_lag_sun:.2f}h)")
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Lag (hours)")
plt.ylabel("Correlation Coefficient")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Fit lunar model with phase from ephem
def lunar_model_with_ephem_phase(t, amplitude, offset):
    """
    Lunar model with phase determined by ephem positions
    
    Parameters:
    t: time points
    amplitude: scaling factor
    offset: vertical offset
    """
    # Use the raw gravitational model and apply scaling
    return amplitude * raw_moon_grav + offset

# Perform the fit with two parameters
try:
    params, _ = curve_fit(lunar_model_with_ephem_phase, time_hours, lunar_component, p0=[1.0, 0.0])
    fitted_lunar = lunar_model_with_ephem_phase(time_hours, *params)
    
    # Calculate fit quality (R-squared)
    ss_tot = np.sum((lunar_component - np.mean(lunar_component))**2)
    ss_res = np.sum((lunar_component - fitted_lunar)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Lunar model fit with ephem phase:")
    print(f"  Amplitude: {params[0]:.6f}")
    print(f"  Offset: {params[1]:.6f}")
    print(f"  R-squared: {r_squared:.4f}")
    
except Exception as e:
    print(f"Error fitting lunar model with ephem phase: {e}")

# Compare with the original sine model fit from earlier in code
def compare_models():
    """Compare the ephem-based model with the original sine model"""
    plt.figure(figsize=(15, 8))
    plt.title("Comparison of Models")
    
    # Plot lunar component
    plt.plot(datetime_data, lunar_component, 'bo-', label="Lunar Component", alpha=0.5)
    
    # Plot original sine model if available
    try:
        # This uses the fitted_lunar from earlier
        plt.plot(datetime_data, fitted_lunar, 'g-', linewidth=2, label="Original Sine Model")
    except:
        print("Original fitted model not available for comparison")
    
    # Plot ephem-based model
    try:
        ephem_model = lunar_model_with_ephem_phase(time_hours, *params)
        plt.plot(datetime_data, ephem_model, 'r-', linewidth=2, label="Ephem-based Model")
    except:
        print("Ephem-based model not available for comparison")
    
    # Add a phase-shifted ephem model for optimal fit (at the lag identified by cross-correlation)
    try:
        # Create lagged indices for resampling
        optimal_lag_samples = int(max_lag_moon / avg_hours_per_sample)
        if optimal_lag_samples != 0:
            if optimal_lag_samples > 0:
                shifted_moon_grav = np.concatenate([moon_grav[optimal_lag_samples:], np.zeros(optimal_lag_samples)])
            else:
                shifted_moon_grav = np.concatenate([np.zeros(-optimal_lag_samples), moon_grav[:optimal_lag_samples]])
                
            # Scale to match lunar component
            shifted_model = (shifted_moon_grav - np.mean(shifted_moon_grav)) / np.std(shifted_moon_grav) * np.std(lunar_component) + np.mean(lunar_component)
            plt.plot(datetime_data, shifted_model, 'm--', linewidth=2, label=f"Phase-shifted Model (lag: {max_lag_moon:.2f}h)")
    except Exception as e:
        print(f"Error creating phase-shifted model: {e}")
    
    plt.xlabel("Date and Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.tight_layout()

# Run the model comparison
compare_models()

plt.show()


def combined_lunar_model(t, amplitude, phase_shift, vertical_offset, invert=False):
    """
    More comprehensive lunar model with phase shift and option to invert
    
    Parameters:
    t: time points
    amplitude: scaling factor
    phase_shift: time lag in hours
    vertical_offset: vertical shift of the signal
    invert: whether to invert the relationship
    """
    # Convert phase shift from hours to number of samples
    shift_samples = int(phase_shift / avg_hours_per_sample)
    
    # Create shifted moon gravity array
    n = len(moon_grav)
    if shift_samples >= 0:
        # Positive shift (model lags behind data)
        shifted = np.concatenate([moon_grav[shift_samples:], moon_grav[:shift_samples]])
    else:
        # Negative shift (model ahead of data)
        shift_samples = abs(shift_samples)
        shifted = np.concatenate([moon_grav[-shift_samples:], moon_grav[:-shift_samples]])
    
    # Normalise
    normalised = (shifted - np.mean(shifted)) / np.std(shifted)
    
    # Apply inversion if needed
    if invert:
        normalised = -normalised
    
    # Scale and offset
    return amplitude * normalised + vertical_offset

# Determine whether to invert based on correlation
invert_moon = max_corr_moon < 0
initial_phase_shift = max_lag_moon

# Fit the improved model
try:
    params, _ = curve_fit(
        lambda t, amp, phase, offset: combined_lunar_model(t, amp, phase, offset, invert=invert_moon),
        time_hours, 
        lunar_component,
        p0=[np.std(lunar_component), initial_phase_shift, np.mean(lunar_component)]
    )
    
    fitted_model = combined_lunar_model(
        time_hours, params[0], params[1], params[2], invert=invert_moon
    )
    
    # Calculate fit quality
    ss_tot = np.sum((lunar_component - np.mean(lunar_component))**2)
    ss_res = np.sum((lunar_component - fitted_model)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nComprehensive Lunar Model Results:")
    print(f"  Amplitude: {params[0]:.6f}")
    print(f"  Phase Shift: {params[1]:.2f} hours")
    print(f"  Vertical Offset: {params[2]:.6f}")
    print(f"  Inverted: {invert_moon}")
    print(f"  R-squared: {r_squared:.4f}")
    
    # Plot the improved fit
    plt.figure(figsize=(15, 6))
    plt.title(f"Comprehensive Lunar Model (R² = {r_squared:.4f})")
    plt.plot(datetime_data, lunar_component, 'bo-', label="Lunar Component", alpha=0.7)
    plt.plot(datetime_data, fitted_model, 'r-', linewidth=2, 
             label=f" Model ")
    plt.xlabel("Date and Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    
except Exception as e:
    print(f"Error fitting comprehensive model: {e}")
    
    
 # Create a function to analyze and visualize sun's phase relationship with data
def analyse_sun_phase():
    """
    Create more detailed analysis of the sun's phase relationship with the interferometer data
    """
    print("Analysing sun phase relationship...")
    
    # Calculate sun positions and parameters
    sun_alt = []
    sun_az = []
    sun_distance = []
    
    for dt in datetime_data:
        observer.date = ephem.Date(dt)
        sun = ephem.Sun(observer)
        sun_alt.append(float(sun.alt))
        sun_az.append(float(sun.az))
        sun_distance.append(float(sun.earth_distance))
    
    # Convert to numpy arrays
    sun_alt = np.array(sun_alt)
    sun_az = np.array(sun_az)
    sun_distance = np.array(sun_distance)
    
    # Calculate day/night cycle
    is_daytime = sun_alt > 0
    
    # Create a more detailed sun model including altitude and azimuth components
    def detailed_sun_model(t, alt_weight, az_weight, dist_weight, offset):
        """
        More sophisticated sun model incorporating altitude, azimuth, and distance
        """
        # Normalise each component
        norm_alt = (sun_alt - np.mean(sun_alt)) / np.std(sun_alt)
        norm_az = np.sin(sun_az)  # Use sine of azimuth for periodicity
        norm_dist = (sun_distance - np.mean(sun_distance)) / np.std(sun_distance)
        
        # Combine components with weights
        model = (alt_weight * norm_alt + 
                 az_weight * norm_az + 
                 dist_weight * norm_dist)
        
        # Normalise the combined model
        model_norm = (model - np.mean(model)) / np.std(model) * np.std(lunar_component)
        
        return model_norm + offset
    
    # Fit the detailed sun model
    try:
        # Initial parameters: equal weights and zero offset
        initial_params = [0.5, 0.3, 0.2, np.mean(lunar_component)]
        
        sun_params, _ = curve_fit(
            detailed_sun_model,
            time_hours, 
            lunar_component,
            p0=initial_params
        )
        
        fitted_sun_model = detailed_sun_model(time_hours, *sun_params)
        
        # Calculate fit quality
        ss_tot = np.sum((lunar_component - np.mean(lunar_component))**2)
        ss_res = np.sum((lunar_component - fitted_sun_model)**2)
        sun_r_squared = 1 - (ss_res / ss_tot)
        
        print(f"Detailed Sun Model Results:")
        print(f"  Altitude Weight: {sun_params[0]:.4f}")
        print(f"  Azimuth Weight: {sun_params[1]:.4f}")
        print(f"  Distance Weight: {sun_params[2]:.4f}")
        print(f"  Offset: {sun_params[3]:.6f}")
        print(f"  R-squared: {sun_r_squared:.4f}")
    except Exception as e:
        print(f"Error fitting detailed sun model: {e}")
        sun_r_squared = 0
        fitted_sun_model = np.zeros_like(lunar_component)
    
    # Create combined visualisation
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original data with sun model
    plt.subplot(3, 1, 1)
    plt.title(f"Interferometer Data with Sun Model (R² = {sun_r_squared:.4f})")
    plt.plot(datetime_data, lunar_component, 'bo-', alpha=0.6, label="Lunar Component")
    plt.plot(datetime_data, fitted_sun_model, 'orange', linewidth=2, label="Sun Model")
    
    # Add day/night shading
    for i in range(len(datetime_data)-1):
        if is_daytime[i]:
            plt.axvspan(datetime_data[i], datetime_data[i+1], alpha=0.1, color='yellow')
    
    plt.xlabel("Date and Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Plot 2: Sun altitude vs amplitude
    plt.subplot(3, 1, 2)
    plt.title("Sun Position Components vs Time")
    plt.plot(datetime_data, sun_alt, 'r-', label="Sun Altitude")
    
    # Convert azimuth to 0-1 range for better visualisation
    norm_az = np.array(sun_az) / (2*np.pi)
    plt.plot(datetime_data, norm_az, 'g-', label="Sun Azimuth (normalized)")
    
    # Add day/night shading
    for i in range(len(datetime_data)-1):
        if is_daytime[i]:
            plt.axvspan(datetime_data[i], datetime_data[i+1], alpha=0.1, color='yellow')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Date and Time")
    plt.ylabel("Position (radians / normalized)")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Plot 3: Scatter plot of sun altitude vs interferometer amplitude
    plt.subplot(3, 1, 3)
    plt.title("Sun Altitude vs Interferometer Amplitude")
    
    # Color points by daytime/nighttime
    colors = ['gold' if day else 'navy' for day in is_daytime]
    
    # Create scatter plot
    plt.scatter(sun_alt, lunar_component, c=colors, alpha=0.7)
    
    # Add regression line
    z = np.polyfit(sun_alt, lunar_component, 1)
    p = np.poly1d(z)
    sun_alt_sorted = np.sort(sun_alt)
    plt.plot(sun_alt_sorted, p(sun_alt_sorted), "r--", alpha=0.8)
    
    # Add correlation coefficient
    corr_coef = np.corrcoef(sun_alt, lunar_component)[0, 1]
    plt.annotate(f"Correlation: {corr_coef:.4f}", xy=(0.05, 0.95), 
                xycoords='axes fraction', fontsize=10)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel("Sun Altitude (radians)")
    plt.ylabel("Interferometer Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return sun_r_squared, fitted_sun_model

# Comparison of moon and sun contributions
def compare_celestial_contributions(moon_model, sun_model, 
                                   moon_r_squared, sun_r_squared):
    """
    Compare and visualize the relative contributions of moon and sun
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Moon vs Sun model contributions
    plt.subplot(2, 1, 1)
    plt.title("Moon vs Sun Model Contributions")
    plt.plot(datetime_data, lunar_component, 'ko-', alpha=0.5, label="Data")
    plt.plot(datetime_data, moon_model, 'b-', linewidth=2, 
             label=f"Moon Model (R² = {moon_r_squared:.4f})")
    plt.plot(datetime_data, sun_model, 'orange', linewidth=2, 
             label=f"Sun Model (R² = {sun_r_squared:.4f})")
    
    # Combined model (weighted average based on R-squared)
    if moon_r_squared + sun_r_squared > 0:
        moon_weight = moon_r_squared / (moon_r_squared + sun_r_squared)
        sun_weight = sun_r_squared / (moon_r_squared + sun_r_squared)
    else:
        moon_weight = 0.5
        sun_weight = 0.5
    
    combined_model = moon_weight * moon_model + sun_weight * sun_model
    
    # Calculate R-squared for combined model
    ss_tot = np.sum((lunar_component - np.mean(lunar_component))**2)
    ss_res = np.sum((lunar_component - combined_model)**2)
    combined_r_squared = 1 - (ss_res / ss_tot)
    
    plt.plot(datetime_data, combined_model, 'g-', linewidth=2, 
             label=f"Combined Model (R² = {combined_r_squared:.4f})")
    
    plt.xlabel("Date and Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    # Plot 2: Residuals after removing celestial influences
    plt.subplot(2, 1, 2)
    plt.title("Residuals After Removing Celestial Influences")
    
    residuals = lunar_component - combined_model
    plt.plot(datetime_data, residuals, 'ro-', alpha=0.6)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Add smoothed trend line to check for remaining patterns
    try:
        window = min(15, len(residuals) // 3 * 2 + 1)  # Ensure window is odd and not too large
        if window > 3:
            smoothed = savgol_filter(residuals, window, 2)
            plt.plot(datetime_data, smoothed, 'b-', linewidth=2, 
                    label="Smoothed Trend")
            plt.legend()
    except Exception as e:
        print(f"Error smoothing residuals: {e}")
    
    plt.xlabel("Date and Time")
    plt.ylabel("Residual Amplitude")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    
    plt.tight_layout()
    plt.show()
    
    return combined_r_squared, combined_model

# Perform complete phase analysis
def perform_complete_phase_analysis():
    """
    Perform a comprehensive phase analysis including both moon and sun
    """
    print("\nPerforming comprehensive phase analysis...")
    
    # Run sun phase analysis
    sun_r_squared, fitted_sun_model = analyse_sun_phase()
    
    # Use the comprehensive lunar model results from previous analysis
    moon_r_squared = r_squared  # From your existing analysis
    fitted_moon_model = fitted_model  # From your existing analysis
    
    # Compare contributions
    combined_r_squared, combined_model = compare_celestial_contributions(
        fitted_moon_model, fitted_sun_model, moon_r_squared, sun_r_squared)
    
    # Calculate relative contributions
    if moon_r_squared + sun_r_squared > 0:
        moon_contribution = moon_r_squared / (moon_r_squared + sun_r_squared) * 100
        sun_contribution = sun_r_squared / (moon_r_squared + sun_r_squared) * 100
    else:
        moon_contribution = 50
        sun_contribution = 50
    
    print("\nRelative Contributions to Interferometer Phase:")
    print(f"  Moon: {moon_contribution:.1f}%")
    print(f"  Sun:  {sun_contribution:.1f}%")
    print(f"  Combined model fit (R²): {combined_r_squared:.4f}")
    
    # Create final visualization summarizing findings
    plt.figure(figsize=(10, 6))
    plt.title("Relative Celestial Influences on Interferometer Phase")
    
    # Create pie chart of contributions
    plt.pie([moon_contribution, sun_contribution], 
            labels=['Moon', 'Sun'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['royalblue', 'orange'],
            explode=(0.1, 0),
            shadow=True)
    
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.show()
    
    return moon_contribution, sun_contribution, combined_r_squared

# Call the functions to perform analysis
perform_complete_phase_analysis()   