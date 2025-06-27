"""
CUDA acceleration utilities for YAAPT pitch tracking algorithm.

This module provides GPU-accelerated implementations of the computationally
intensive operations in the YAAPT pitch tracking algorithm using CUDA via Numba.

Version 1.0.12
"""

import numpy as np
from numba import cuda, float32, float64, complex64, complex128, jit
import math
import scipy.signal as signal

# Check if CUDA is available
CUDA_AVAILABLE = False
try:
    CUDA_AVAILABLE = cuda.is_available()
except:
    CUDA_AVAILABLE = False

def get_cuda_info():
    """Return information about CUDA availability and devices."""
    if not CUDA_AVAILABLE:
        return {"available": False, "reason": "CUDA not available or not installed"}
    
    try:
        devices = []
        for i in range(cuda.get_device_count()):
            device = cuda.get_device(i)
            device_info = {
                "id": i,
                "name": device.name,
                "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
                "compute_capability": device.compute_capability,
                "total_memory": device.total_memory
            }
            devices.append(device_info)
        
        return {"available": True, "devices": devices}
    except Exception as e:
        return {"available": False, "reason": str(e)}

# CUDA kernels for window functions

@cuda.jit
def cuda_apply_window(data_matrix, window, result):
    """Apply window function to data matrix."""
    i, j = cuda.grid(2)
    if i < result.shape[0] and j < result.shape[1]:
        result[i, j] = data_matrix[i, j] * window[j]

# CUDA implementation of FFT operations
def cuda_rfft(data_matrix, nfft):
    """CUDA-accelerated FFT for real input."""
    if not CUDA_AVAILABLE:
        return np.fft.rfft(data_matrix, nfft)
    
    try:
        # Note: Numba doesn't directly support FFT operations yet
        # We'll use numpy's FFT and accelerate other operations
        return np.fft.rfft(data_matrix, nfft)
    except:
        return np.fft.rfft(data_matrix, nfft)

# CUDA implementation of SHC (Spectral Harmonic Correlation) operations
@cuda.jit
def cuda_shc_kernel(spectrogram, nfreqs, num_harmonics, result):
    """Kernel for Spectral Harmonic Correlation calculation."""
    i, j = cuda.grid(2)
    if i < spectrogram.shape[0] and j < nfreqs:
        # Base frequency index
        base_idx = j
        
        # Calculate SHC for this frequency
        shc_value = 0.0
        for h in range(1, num_harmonics + 1):
            harm_idx = base_idx * h
            if harm_idx < spectrogram.shape[1]:
                shc_value += spectrogram[i, harm_idx]
        
        # Store result
        result[i, j] = shc_value

def cuda_compute_shc(spectrogram, nfreqs, num_harmonics):
    """CUDA-accelerated SHC computation."""
    if not CUDA_AVAILABLE:
        # Fallback to CPU implementation
        result = np.zeros((spectrogram.shape[0], nfreqs))
        for i in range(spectrogram.shape[0]):
            for j in range(nfreqs):
                for h in range(1, num_harmonics + 1):
                    harm_idx = j * h
                    if harm_idx < spectrogram.shape[1]:
                        result[i, j] += spectrogram[i, harm_idx]
        return result
    
    try:
        # Allocate memory on device
        d_result = cuda.device_array((spectrogram.shape[0], nfreqs))
        d_spectrogram = cuda.to_device(spectrogram)
        
        # Configure kernel
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(spectrogram.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(nfreqs / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Execute kernel
        cuda_shc_kernel[blockspergrid, threadsperblock](
            d_spectrogram, nfreqs, num_harmonics, d_result)
        
        # Transfer result back to host
        return d_result.copy_to_host()
    except:
        # Fallback to CPU implementation
        result = np.zeros((spectrogram.shape[0], nfreqs))
        for i in range(spectrogram.shape[0]):
            for j in range(nfreqs):
                for h in range(1, num_harmonics + 1):
                    harm_idx = j * h
                    if harm_idx < spectrogram.shape[1]:
                        result[i, j] += spectrogram[i, harm_idx]
        return result

# CUDA implementation of NCCF (Normalized Cross-Correlation Function)
@cuda.jit
def cuda_nccf_kernel(signal, lag_min, lag_max, frame_len, result):
    """Kernel for Normalized Cross-Correlation calculation."""
    frame_idx, lag = cuda.grid(2)
    if frame_idx < result.shape[0] and lag < (lag_max - lag_min):
        actual_lag = lag + lag_min
        numerator = 0.0
        denominator1 = 0.0
        denominator2 = 0.0
        
        # Calculate correlation for this frame and lag
        start_idx = frame_idx * frame_len
        for i in range(frame_len - actual_lag):
            s1 = signal[start_idx + i]
            s2 = signal[start_idx + i + actual_lag]
            numerator += s1 * s2
            denominator1 += s1 * s1
            denominator2 += s2 * s2
        
        # Normalize
        if denominator1 > 0 and denominator2 > 0:
            result[frame_idx, lag] = numerator / math.sqrt(denominator1 * denominator2)
        else:
            result[frame_idx, lag] = 0.0

def cuda_compute_nccf(signal, nframes, frame_len, frame_jump, lag_min, lag_max):
    """CUDA-accelerated NCCF computation."""
    if not CUDA_AVAILABLE:
        # Fallback to CPU implementation
        return cpu_compute_nccf(signal, nframes, frame_len, frame_jump, lag_min, lag_max)
    
    try:
        # Allocate memory on device
        result = np.zeros((nframes, lag_max - lag_min))
        d_result = cuda.to_device(result)
        d_signal = cuda.to_device(signal)
        
        # Configure kernel
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(nframes / threadsperblock[0])
        blockspergrid_y = math.ceil((lag_max - lag_min) / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Execute kernel
        cuda_nccf_kernel[blockspergrid, threadsperblock](
            d_signal, lag_min, lag_max, frame_len, d_result)
        
        # Transfer result back to host
        return d_result.copy_to_host()
    except:
        # Fallback to CPU implementation
        return cpu_compute_nccf(signal, nframes, frame_len, frame_jump, lag_min, lag_max)

# CPU version of NCCF for fallback
def cpu_compute_nccf(signal, nframes, frame_len, frame_jump, lag_min, lag_max):
    """CPU implementation of NCCF computation."""
    result = np.zeros((nframes, lag_max - lag_min))
    
    for frame_idx in range(nframes):
        start_idx = frame_idx * frame_jump
        frame_end = start_idx + frame_len
        
        for lag in range(lag_min, lag_max):
            if frame_end + lag > len(signal):
                continue
                
            s1 = signal[start_idx:frame_end-lag]
            s2 = signal[start_idx+lag:frame_end]
            
            numerator = np.sum(s1 * s2)
            denominator = np.sqrt(np.sum(s1 * s1) * np.sum(s2 * s2))
            
            if denominator > 0:
                result[frame_idx, lag-lag_min] = numerator / denominator
    
    return result

# CUDA implementation of dynamic programming for pitch tracking
@jit(nopython=True)
def numba_dynamic_programming(merit_function, transition_cost, w4):
    """Numba-accelerated dynamic programming for pitch tracking."""
    num_frames, num_candidates = merit_function.shape
    
    # Initialize cost matrix and backtracking indices
    cost = np.zeros((num_frames, num_candidates))
    prev = np.zeros((num_frames, num_candidates), dtype=np.int32)
    
    # Set costs for first frame
    cost[0, :] = merit_function[0, :]
    
    # Forward pass
    for i in range(1, num_frames):
        for j in range(num_candidates):
            # Find minimum cost path to current state
            min_cost = float('inf')
            min_idx = 0
            
            for k in range(num_candidates):
                # Calculate transition cost
                trans = transition_cost[k, j]
                # Calculate total cost
                total_cost = cost[i-1, k] + trans
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    min_idx = k
            
            # Store minimum cost path
            cost[i, j] = merit_function[i, j] * w4 + min_cost
            prev[i, j] = min_idx
    
    # Backtracking
    path = np.zeros(num_frames, dtype=np.int32)
    path[-1] = np.argmin(cost[-1, :])
    
    for i in range(num_frames-2, -1, -1):
        path[i] = prev[i+1, path[i+1]]
    
    return path

def cuda_dynamic_programming(merit_function, transition_cost, w4):
    """CUDA-accelerated dynamic programming for pitch tracking."""
    # Currently using Numba JIT acceleration since dynamic programming 
    # is challenging to efficiently parallelize on GPU
    return numba_dynamic_programming(merit_function, transition_cost, w4)