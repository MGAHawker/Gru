# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 01:28:06 2025

@author: migsh
"""

import numpy as np
import matplotlib.pyplot as plt
import optoanalysis as opt
import glob
import os
import datetime
import plotly.express as px
import plotly.io as pio

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
file_datetimes = [(file_path, datetime.datetime.fromtimestamp(os.path.getmtime(file_path)))
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
downsampling_factor = 5  # Adjust as needed but will affect sampling freq
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
downsampling_factor = 5  # Adjust as needed but will effect sampling freq 
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
    bin_duration (float): Duration of each bin in seconds (default 30 minutes)
    
    Returns:
    Tuple of downsampled time and voltage arrays
    """
    valid_time = []
    valid_voltage = []
    
    for i, (time_data, voltage_data) in enumerate(zip(time_segments, voltage_segments)):
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
    
    # Initialize arrays to store downsampled data
    downsampled_time = []
    downsampled_voltage = []
    
    # Downsample for each time bin
    for i in range(len(time_bins) - 1):
        # Create a mask for data within the current time bin
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

def create_interactive_plot(time_segments, voltage_segments, gap_indices):
    """
    Create an interactive Plotly Express plot with downsampled data
    """
    # Perform time-based downsampling
    downsampled_time, downsampled_voltage = time_based_downsampling(
        time_segments, voltage_segments, gap_indices, bin_duration=1800
    )
    
    # Create interactive plot using Plotly Express
    fig = px.line(
        x=downsampled_time, 
        y=downsampled_voltage,
        title='Downsampled DC Trace (30-minute bins)',
        labels={'x': 'Time (seconds)', 'y': 'Average Voltage (V)'}
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate='Time: %{x:.2f} seconds<br>Voltage: %{y:.4f} V<extra></extra>'
    )
    
    # Customize layout
    fig.update_layout(
        height=600,
        width=1000,
        hovermode='closest'
    )
    
    # Return the figure object
    return fig

# Create and save interactive Plotly plot
fig = create_interactive_plot(ordered_time_segments, ordered_data_segments, downsampled_gap_indices)
# Save interactive plot as HTML
print('Making interactive plot')
pio.write_html(fig, file="voltage_data_interactive.html")