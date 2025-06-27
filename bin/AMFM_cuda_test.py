#!/usr/bin/env python

"""
Script to test the CUDA-accelerated AMFM_decompy package.

This script compares the performance of CPU vs GPU implementations
for both QHM and YAAPT algorithms.

Version 1.0.12
"""
import amfm_decompy
import amfm_decompy.pYAAPT as pyaapt
import amfm_decompy.pyQHM as pyqhm
import amfm_decompy.basic_tools as basic
import os.path
import time
import numpy as np
import os

def main():
    # Display CUDA information
    print("\nCUDA Information:")
    print("-" * 50)
    cuda_available = pyqhm.is_cuda_available() and pyaapt.is_cuda_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        cuda_info = pyqhm.cuda_info()
        print("CUDA Devices:")
        for device in cuda_info.get("devices", []):
            print(f"  - Device {device['id']}: {device['name']}")
            print(f"    Compute capability: {device['compute_capability']}")
            print(f"    Total memory: {device['total_memory'] / (1024**2):.2f} MB")
    print("-" * 50)
    
    # Set environment variables to control CUDA usage
    os.environ["AMFM_USE_CUDA"] = "1"  # Enable CUDA
    
    # Declare the variables
    file_name = os.path.dirname(amfm_decompy.__file__)+os.sep+"sample.wav"
    window_duration = 0.015   # in seconds
    nharm_max = 25
    
    # Performance comparison for both algorithms
    print("\nPerformance Comparison: CPU vs GPU")
    print("-" * 50)
    
    # Test YAAPT pitch tracking
    print("\n1. Testing YAAPT pitch tracking")
    run_yaapt_comparison(file_name)
    
    # Test QHM decomposition
    print("\n2. Testing QHM decomposition")
    run_qhm_comparison(file_name, window_duration, nharm_max)

def run_yaapt_comparison(file_name):
    """Run performance comparison for YAAPT algorithm."""
    # Create the signal object
    signal = basic.SignalObj(file_name)
    
    # Run CPU version
    os.environ["AMFM_USE_CUDA"] = "0"
    start_time = time.time()
    pitch_cpu = pyaapt.yaapt(signal)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # Run GPU version if available
    if pyaapt.is_cuda_available():
        os.environ["AMFM_USE_CUDA"] = "1"
        start_time = time.time()
        pitch_gpu = pyaapt.yaapt(signal)
        gpu_time = time.time() - start_time
        print(f"GPU Time: {gpu_time:.4f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results similarity
        # Calculate mean absolute difference in pitch values
        nonzero_indices = np.logical_and(pitch_cpu.values > 0, pitch_gpu.values > 0)
        if np.any(nonzero_indices):
            pitch_diff = np.abs(pitch_cpu.values[nonzero_indices] - pitch_gpu.values[nonzero_indices])
            mean_diff = np.mean(pitch_diff)
            max_diff = np.max(pitch_diff)
            print(f"Mean Pitch Difference: {mean_diff:.6f} Hz")
            print(f"Max Pitch Difference: {max_diff:.6f} Hz")
        else:
            print("No voiced frames detected for comparison")
    else:
        print("CUDA is not available for YAAPT. Skipping GPU test.")
    
def run_qhm_comparison(file_name, window_duration, nharm_max):
    """Run performance comparison for QHM algorithm."""
    # Create the signal object
    signal = basic.SignalObj(file_name)
    
    # Create the window object
    window = pyqhm.SampleWindow(window_duration, signal.fs)
    
    # Create the pitch object and calculate its attributes
    pitch = pyaapt.yaapt(signal)
    
    # Set the number of modulated components
    signal.set_nharm(pitch.values, nharm_max)
    
    # Run CPU version
    os.environ["AMFM_USE_CUDA"] = "0"
    start_time = time.time()
    QHM_cpu = pyqhm.qhm(signal, pitch, window, 0.001, N_iter=3, phase_tech='phase')
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds, SRER: {QHM_cpu.SRER:.4f} dB")
    
    # Run GPU version if available
    if pyqhm.is_cuda_available():
        os.environ["AMFM_USE_CUDA"] = "1"
        start_time = time.time()
        QHM_gpu = pyqhm.qhm(signal, pitch, window, 0.001, N_iter=3, phase_tech='phase')
        gpu_time = time.time() - start_time
        print(f"GPU Time: {gpu_time:.4f} seconds, SRER: {QHM_gpu.SRER:.4f} dB")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results are similar
        print(f"SRER Difference: {abs(QHM_cpu.SRER - QHM_gpu.SRER):.6f} dB")
    else:
        print("CUDA is not available for QHM. Skipping GPU test.")

    # Run aQHM and eaQHM if time allows
    if pyqhm.is_cuda_available():
        print("\nTesting aQHM")
        # CPU - aQHM
        os.environ["AMFM_USE_CUDA"] = "0"
        start_time = time.time()
        aQHM_cpu = pyqhm.aqhm(signal, QHM_cpu, pitch, window, 0.001, N_iter=3, N_runs=2, phase_tech='phase')
        cpu_time = time.time() - start_time
        print(f"CPU Time: {cpu_time:.4f} seconds, SRER: {aQHM_cpu.SRER:.4f} dB")
        
        # GPU - aQHM
        os.environ["AMFM_USE_CUDA"] = "1"
        start_time = time.time()
        aQHM_gpu = pyqhm.aqhm(signal, QHM_gpu, pitch, window, 0.001, N_iter=3, N_runs=2, phase_tech='phase')
        gpu_time = time.time() - start_time
        print(f"GPU Time: {gpu_time:.4f} seconds, SRER: {aQHM_gpu.SRER:.4f} dB")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        print("\nTesting eaQHM")
        # CPU - eaQHM
        os.environ["AMFM_USE_CUDA"] = "0"
        start_time = time.time()
        eaQHM_cpu = pyqhm.eaqhm(signal, aQHM_cpu, pitch, window, 0.001, N_iter=3, N_runs=2, phase_tech='phase')
        cpu_time = time.time() - start_time
        print(f"CPU Time: {cpu_time:.4f} seconds, SRER: {eaQHM_cpu.SRER:.4f} dB")
        
        # GPU - eaQHM
        os.environ["AMFM_USE_CUDA"] = "1"
        start_time = time.time()
        eaQHM_gpu = pyqhm.eaqhm(signal, aQHM_gpu, pitch, window, 0.001, N_iter=3, N_runs=2, phase_tech='phase')
        gpu_time = time.time() - start_time
        print(f"GPU Time: {gpu_time:.4f} seconds, SRER: {eaQHM_gpu.SRER:.4f} dB")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()