import numpy as np
from numba import jit, prange
from typing import Union, Tuple

@jit(nopython=True, parallel=True)
def exp(x: np.ndarray) -> np.ndarray:
    return np.exp(x)

@jit(nopython=True, parallel=True)
def log(x: np.ndarray) -> np.ndarray:
    return np.log(x)

@jit(nopython=True, parallel=True)
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

@jit(nopython=True, parallel=True)
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

@jit(nopython=True, parallel=True)
def clip(x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    return np.clip(x, min_val, max_val)

@jit(nopython=True, parallel=True)
def mean(x: np.ndarray) -> float:
    return np.mean(x)

@jit(nopython=True, parallel=True)
def sum(x: np.ndarray) -> float:
    return np.sum(x)

@jit(nopython=True, parallel=True)
def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)

@jit(nopython=True, parallel=True)
def element_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

@jit(nopython=True, parallel=True)
def element_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / b

@jit(nopython=True)
def broadcast_to(arr: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(arr, shape)

@jit(nopython=True, parallel=True)
def conv2d(x: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int], padding: Tuple[int, int]) -> np.ndarray:
    N, C, H, W = x.shape
    out_channels, in_channels, kH, kW = kernel.shape

    H_out = (H + 2*padding[0] - kH) // stride[0] + 1
    W_out = (W + 2*padding[1] - kW) // stride[1] + 1

    out = np.zeros((N, out_channels, H_out, W_out))

    if padding[0] > 0 or padding[1] > 0:
        x_padded = np.pad(x, ((0,0), (0,0), (padding[0],padding[0]), (padding[1],padding[1])), mode='constant')
    else:
        x_padded = x

    for n in prange(N):
        for c_out in range(out_channels):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride[0]
                    w_start = w * stride[1]
                    h_end = h_start + kH
                    w_end = w_start + kW
                    
                    receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, c_out, h, w] = np.sum(receptive_field * kernel[c_out])
    
    return out

@jit(nopython=True, parallel=True)
def conv2d_backward_x(grad_output: np.ndarray, kernel: np.ndarray, x_shape: Tuple[int, ...],
                     stride: Tuple[int, int], padding: Tuple[int, int]) -> np.ndarray:
    N, C, H, W = x_shape
    out_channels, in_channels, kH, kW = kernel.shape
    _, _, H_out, W_out = grad_output.shape
    
    dx = np.zeros(x_shape)
    
    for n in prange(N):
        for c_in in range(in_channels):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride[0]
                    w_start = w * stride[1]
                    h_end = h_start + kH
                    w_end = w_start + kW
                    
                    for c_out in range(out_channels):
                        dx[n, c_in, h_start:h_end, w_start:w_end] += \
                            kernel[c_out, c_in] * grad_output[n, c_out, h, w]
    
    return dx

@jit(nopython=True, parallel=True)
def conv2d_backward_kernel(grad_output: np.ndarray, x: np.ndarray, kernel_shape: Tuple[int, ...],
                          stride: Tuple[int, int], padding: Tuple[int, int]) -> np.ndarray:
    N, C, H, W = x.shape
    out_channels, in_channels, kH, kW = kernel_shape
    _, _, H_out, W_out = grad_output.shape
    
    dkernel = np.zeros(kernel_shape)

    if padding[0] > 0 or padding[1] > 0:
        x_padded = np.pad(x, ((0,0), (0,0), (padding[0],padding[0]), (padding[1],padding[1])), mode='constant')
    else:
        x_padded = x
    
    for c_out in prange(out_channels):
        for c_in in range(in_channels):
            for h in range(H_out):
                for w in range(W_out):
                    h_start = h * stride[0]
                    w_start = w * stride[1]
                    h_end = h_start + kH
                    w_end = w_start + kW
                    
                    for n in range(N):
                        dkernel[c_out, c_in] += \
                            x_padded[n, c_in, h_start:h_end, w_start:w_end] * grad_output[n, c_out, h, w]
    
    return dkernel 