import numpy as np
from typing import Union, Tuple, Optional
from . import jit_ops

class CPUOps:
    """CPU-based operations accelerated with Numba JIT compilation."""
    
    def exp(self, x: np.ndarray) -> np.ndarray:
        return jit_ops.exp(x)
    
    def log(self, x: np.ndarray) -> np.ndarray:
        return jit_ops.log(x)
    
    def abs(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)
    
    def sign(self, x: np.ndarray) -> np.ndarray:
        return np.sign(x)
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return jit_ops.tanh(x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return jit_ops.relu(x)
    
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        return jit_ops.clip(x, min_val, max_val)
    
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(x)
    
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return jit_ops.element_multiply(a, b)
    
    def divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return jit_ops.element_divide(a, b)
    
    def sum(self, x: np.ndarray) -> float:
        return jit_ops.sum(x)
    
    def mean(self, x: np.ndarray) -> float:
        return jit_ops.mean(x)
    
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype=np.float32) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype=np.float32) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        return np.ones(shape, dtype=dtype)
    
    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)
    
    def ones_like(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return jit_ops.matmul(a, b)
    
    def conv2d(self, x: np.ndarray, kernel: np.ndarray, stride: Tuple[int, int], 
               padding: Tuple[int, int]) -> np.ndarray:
        return jit_ops.conv2d(x, kernel, stride, padding)
    
    def conv2d_backward_x(self, grad_output: np.ndarray, kernel: np.ndarray, 
                         x_shape: Tuple[int, ...], stride: Tuple[int, int], 
                         padding: Tuple[int, int]) -> np.ndarray:
        return jit_ops.conv2d_backward_x(grad_output, kernel, x_shape, stride, padding)
    
    def conv2d_backward_kernel(self, grad_output: np.ndarray, x: np.ndarray, 
                             kernel_shape: Tuple[int, ...], stride: Tuple[int, int], 
                             padding: Tuple[int, int]) -> np.ndarray:
        return jit_ops.conv2d_backward_kernel(grad_output, x, kernel_shape, stride, padding)

cpu_ops = CPUOps() 