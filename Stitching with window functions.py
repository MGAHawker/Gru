# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:01:14 2025

@author: migsh
"""

import numpy as np
import scipy as scipy
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import optoanalysis as opt
import glob
import os
from datetime import timedelta
import datetime

# Get all the .trc files, manual loading would be ridiculously long
base_dir = "C:/Users/migsh/desktop/12 hour chunk 1"
file_pattern = os.path.join(base_dir, "C2--Trace--*.trc")
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
            baseline_voltage = 0  # or calculate average from segments
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
    
    # Initialize arrays for averaged data and corresponding timestamps
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
    
    # Initialize arrays for averaged data and corresponding timestamps
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

def improved_crossfade(time_segments, data_segments, gap_indices, window_type='tukey', overlap_samples=200, fade_length_ratio=0.2):
    """
    Enhanced crossfading function for smoother transitions between segments.
    
    Parameters:
    time_segments: List of time arrays for each segment
    data_segments: List of data arrays for each segment
    gap_indices: List of indices where significant gaps occur
    window_type: Type of window function to use ('cosine', 'tukey', 'hann', 'blackman')
    overlap_samples: Number of samples to overlap for crossfading
    fade_length_ratio: Ratio of segment length to use for fading (0.0-0.5)
    """
    if not data_segments:
        return np.array([]), np.array([])
    
    # Initialize stitched data with first segment
    stitched_data = data_segments[0].copy()
    stitched_time = time_segments[0].copy()
    
    # Process each subsequent segment
    for i in range(1, len(data_segments)):
        current_segment = data_segments[i]
        current_time = time_segments[i]
        
        # Determine if there's a significant gap
        has_gap = (i-1) in gap_indices
        
        if has_gap:
            # Need to fade out the end of previous segment and fade in the start of current segment
            prev_fade_length = min(len(stitched_data), int(len(current_segment) * fade_length_ratio))
            curr_fade_length = min(len(current_segment), int(len(current_segment) * fade_length_ratio))
            
            # Create window functions for both end of previous segment and start of current segment
            if window_type == 'cosine':
                # Create fade-out window for end of previous segment
                fade_out = np.sin(np.linspace(np.pi/2, 0, prev_fade_length))**2
                # Create fade-in window for start of current segment
                fade_in = np.sin(np.linspace(0, np.pi/2, curr_fade_length))**2
            elif window_type == 'tukey':
                # Create fade-out window for end of previous segment
                fade_out = signal.windows.tukey(2*prev_fade_length, alpha=0.75)[prev_fade_length:]
                # Create fade-in window for start of current segment
                fade_in = signal.windows.tukey(2*curr_fade_length, alpha=0.75)[:curr_fade_length]
            elif window_type == 'hann':
                # Create fade-out window for end of previous segment
                fade_out = signal.windows.hann(2*prev_fade_length)[prev_fade_length:]
                # Create fade-in window for start of current segment
                fade_in = signal.windows.hann(2*curr_fade_length)[:curr_fade_length]
            elif window_type == 'blackman':
                # Create fade-out window for end of previous segment
                fade_out = signal.windows.blackman(2*prev_fade_length)[prev_fade_length:]
                # Create fade-in window for start of current segment
                fade_in = signal.windows.blackman(2*curr_fade_length)[:curr_fade_length]
            else:
                raise ValueError(f"Window type {window_type} not supported")
            
            # Apply fade-out to end of previous stitched data
            if prev_fade_length > 0:
                stitched_data[-prev_fade_length:] *= fade_out
            
            # Apply fade-in to start of current segment
            windowed_segment = current_segment.copy()
            if curr_fade_length > 0:
                windowed_segment[:curr_fade_length] *= fade_in
            
            # Append the windowed segment to the stitched data
            stitched_data = np.append(stitched_data, windowed_segment)
            stitched_time = np.append(stitched_time, current_time)
            
        else:
            # Crossfade segments for smooth transition
            overlap = min(overlap_samples, len(stitched_data), len(current_segment))
            
            # Extract overlap regions
            end_of_prev = stitched_data[-overlap:]
            start_of_curr = current_segment[:overlap]
            
            # Create crossfade transition
            if window_type == 'cosine':
                # Cosine crossfade (equal power crossfade)
                fade_out = np.sin(np.linspace(np.pi/2, 0, overlap))**2
                fade_in = np.sin(np.linspace(0, np.pi/2, overlap))**2
            elif window_type == 'tukey':
                # Tukey window crossfade
                fade_out = signal.windows.tukey(2*overlap, alpha=0.75)[overlap:]
                fade_in = signal.windows.tukey(2*overlap, alpha=0.75)[:overlap]
            elif window_type == 'hann':
                # Hann window crossfade
                fade_out = signal.windows.hann(2*overlap)[overlap:]
                fade_in = signal.windows.hann(2*overlap)[:overlap]
            elif window_type == 'blackman':
                # Blackman window crossfade
                fade_out = signal.windows.blackman(2*overlap)[overlap:]
                fade_in = signal.windows.blackman(2*overlap)[:overlap]
            else:
                raise ValueError(f"Window type {window_type} not supported")
                
            # Apply crossfade
            crossfaded_overlap = end_of_prev * fade_out + start_of_curr * fade_in
            
            # Replace end of previous data with crossfaded section
            stitched_data[-overlap:] = crossfaded_overlap
            
            # Append the rest of the current segment
            stitched_data = np.append(stitched_data, current_segment[overlap:])
            stitched_time = np.append(stitched_time, current_time[overlap:])
    
    return stitched_data, stitched_time

# Call the improved crossfade function with the ordered segments
print("Stitching data segments...")
stitched_data, stitched_time = improved_crossfade(
    ordered_time_segments_downsampled,
    ordered_data_segments_downsampled,
    downsampled_gap_indices,
    window_type='cosine',  # Selected window function
    overlap_samples=40000,  # Not sure what's best yet
    fade_length_ratio= 0.2
)

print(f"Final stitched data: {len(stitched_data)} points")

def visualise_stitching(time_segments, data_segments, stitched_data, stitched_time, gap_indices):
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Original segments with different colors
    plt.subplot(2, 1, 1)
    plt.title("Original Data Segments")
    
    # Plot each segment with a different color
    for i, (time_data, voltage) in enumerate(zip(time_segments, data_segments)):
        is_gap = i in gap_indices
        if is_gap:
            plt.plot(time_data, voltage, 'r--', alpha=0.5, label=f"Gap {i}" if i == gap_indices[0] else "")
        else:
            color = plt.cm.tab10(i % 10)
            plt.plot(time_data, voltage, color=color, alpha=0.7, label=f"Segment {i}")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (V)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Stitched data
    plt.subplot(2, 1, 2)
    plt.title("Stitched Data with Window Function")
    
    plt.plot(stitched_time, stitched_data, 'b-')
    
    # Mark original segment boundaries in stitched data
    for i, segment_time in enumerate(time_segments):
        if i > 0 and i not in gap_indices:  # Skip the first segment and gaps
            plt.axvline(x=segment_time[0], color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (V)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    #plt.savefig("stitched_data_visualisation.png", dpi=300)
    plt.show()
# Call visualisation function
print("Generating visualisation...")
visualise_stitching(ordered_time_segments_downsampled, ordered_data_segments_downsampled, stitched_data, stitched_time, downsampled_gap_indices)
#