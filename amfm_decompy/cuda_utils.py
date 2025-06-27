"""
CUDA acceleration utilities for the AMFM_decompy package.

This module provides GPU-accelerated implementations of the computationally
intensive operations in the AMFM decomposition process using CUDA via Numba.

Version 1.0.12
"""

import numpy as np
from numba import cuda, float32, float64, complex64, complex128
import math

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

# CUDA kernels for matrix operations

@cuda.jit
def cuda_matrix_multiply(A, B, C):
    """Matrix multiplication kernel: C = A * B."""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

@cuda.jit
def cuda_matrix_vector_multiply(A, x, y):
    """Matrix-vector multiplication kernel: y = A * x."""
    i = cuda.grid(1)
    if i < A.shape[0]:
        tmp = 0.0
        for j in range(A.shape[1]):
            tmp += A[i, j] * x[j]
        y[i] = tmp

@cuda.jit
def cuda_complex_exp(arr, out):
    """Complex exponential kernel: out = exp(arr)."""
    i = cuda.grid(1)
    if i < arr.shape[0]:
        # For complex input, calculate exp(a+bi) = exp(a)*(cos(b) + i*sin(b))
        real = arr[i].real
        imag = arr[i].imag
        exp_real = math.exp(real)
        out[i] = complex(exp_real * math.cos(imag), exp_real * math.sin(imag))

def cuda_solve(A, B):
    """Solve the linear system Ax = B using CUDA."""
    # Convert to host arrays to use cuSOLVER
    A_host = A.copy_to_host() if isinstance(A, cuda.devicearray.DeviceNDArray) else A
    B_host = B.copy_to_host() if isinstance(B, cuda.devicearray.DeviceNDArray) else B
    
    # Use NumPy's solver since cuSOLVER requires more complex setup
    x = np.linalg.solve(A_host, B_host)
    
    # Return as device array if needed
    return x

def cuda_exp_matrix(E, freq, window, K):
    """CUDA-accelerated version of the exp_matrix function."""
    if not CUDA_AVAILABLE:
        from amfm_decompy.pyQHM import exp_matrix
        return exp_matrix(E, freq, window, K)
    
    try:
        # Implement the CUDA version of exp_matrix
        # Initial setup
        E_device = cuda.to_device(E)
        freq_device = cuda.to_device(freq)
        
        # Compute exponentials for the upper half
        blocks_per_grid = (E.shape[0] - window.N - 1 + 255) // 256
        threads_per_block = 256
        
        # Calculating phase values for the exponentials
        phase = np.zeros((E.shape[0] - window.N - 1, K), dtype=complex)
        for i in range(K):
            phase[:, i] = 1j * np.pi * 2 * freq[i]
        
        phase_device = cuda.to_device(phase)
        upper_exp_device = cuda.device_array((E.shape[0] - window.N - 1, K), dtype=E.dtype)
        
        # Launch kernel for complex exponential
        cuda_complex_exp[blocks_per_grid, threads_per_block](phase_device, upper_exp_device)
        
        # Copy results back for cumulative product (which is still done on CPU)
        upper_exp = upper_exp_device.copy_to_host()
        
        # Compute cumulative product
        E[window.N+1:, :K] = np.cumprod(upper_exp, axis=0)
        
        # Mirror for the lower half
        E[:window.N, :K] = np.conj(E[window.N+1:, :K][::-1, :])
        
        # Generate the second half of the matrix
        E[:, K:] = E[:, :K] * window.len_vec.reshape(window.length, 1)
        
        return E
        
    except Exception as e:
        # Fall back to CPU implementation on error
        from amfm_decompy.pyQHM import exp_matrix
        return exp_matrix(E, freq, window, K)

def cuda_least_squares(E, E_windowed, windowed_data, window, K):
    """CUDA-accelerated version of the least_squares function."""
    if not CUDA_AVAILABLE:
        from amfm_decompy.pyQHM import least_squares
        return least_squares(E, E_windowed, windowed_data, window, K)
    
    try:
        # Apply window to E
        E_windowed[:, :] = E * window.data.reshape(window.length, 1)
        
        # Transfer data to GPU
        E_windowed_device = cuda.to_device(E_windowed)
        windowed_data_device = cuda.to_device(windowed_data)
        
        # Calculate R = E_windowed.conj().T.dot(E_windowed)
        R_device = cuda.device_array((2*K, 2*K), dtype=complex)
        
        # Grid and block dimensions for matrix multiplication
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(R_device.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(R_device.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Perform matrix multiplication on GPU
        cuda_matrix_multiply[blockspergrid, threadsperblock](
            E_windowed_device.conj().T, E_windowed_device, R_device)
        
        # Calculate B = E_windowed.conj().T.dot(windowed_data)
        B_device = cuda.device_array((2*K, 1), dtype=complex)
        
        # Dimensions for matrix-vector multiplication
        threadsperblock_vec = 256
        blockspergrid_vec = math.ceil(B_device.shape[0] / threadsperblock_vec)
        
        # Perform matrix-vector multiplication on GPU
        cuda_matrix_vector_multiply[blockspergrid_vec, threadsperblock_vec](
            E_windowed_device.conj().T, windowed_data_device, B_device)
        
        # Solve R * coef = B
        coef = cuda_solve(R_device, B_device)
        
        return coef
        
    except Exception as e:
        # Fall back to CPU implementation on error
        from amfm_decompy.pyQHM import least_squares
        return least_squares(E, E_windowed, windowed_data, window, K)