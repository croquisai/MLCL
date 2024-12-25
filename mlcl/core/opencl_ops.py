import pyopencl as cl
import numpy as np
from typing import Union, Tuple, Optional
from .opencl_utils import opencl_manager

class OpenCLOps:
    def __init__(self):
        self.ctx = opencl_manager.ctx
        self.queue = opencl_manager.queue
        self.math_program = opencl_manager.get_program(program_name="math_ops")
        self.reduction_program = opencl_manager.get_program(program_name="reduction_ops")
        self.array_program = opencl_manager.get_program(program_name="array_ops")
    
    def exp(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.exp_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def log(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.log_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def abs(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.abs_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def sign(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.sign_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.tanh_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.relu_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def clip(self, x: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.clip_kernel(self.queue, (x.size,), None, x_buf, out_buf,
                                    np.float32(min_val), np.float32(max_val), np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def sqrt(self, x: np.ndarray) -> np.ndarray:
        output = np.empty_like(x)
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.sqrt_kernel(self.queue, (x.size,), None, x_buf, out_buf, np.uint32(x.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"Arrays must have the same shape. Got {a.shape} and {b.shape}")
        output = np.empty_like(a)
        a_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.multiply_kernel(self.queue, (a.size,), None, a_buf, b_buf, out_buf, np.uint32(a.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def divide(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.shape != b.shape:
            raise ValueError(f"Arrays must have the same shape. Got {a.shape} and {b.shape}")
        output = np.empty_like(a)
        a_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.math_program.divide_kernel(self.queue, (a.size,), None, a_buf, b_buf, out_buf, np.uint32(a.size))
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def sum(self, x: np.ndarray) -> float:
        work_group_size = 256
        n = x.size
        num_groups = (n + work_group_size - 1) // work_group_size
        temp_result = np.empty(num_groups, dtype=np.float32)
        
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        temp_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, temp_result.nbytes)
        
        self.reduction_program.sum_reduce(
            self.queue, (num_groups * work_group_size,), (work_group_size,),
            x_buf, temp_buf, cl.LocalMemory(np.float32(0).nbytes * work_group_size),
            np.int32(n)
        )
        
        cl.enqueue_copy(self.queue, temp_result, temp_buf)
        return float(temp_result.sum())
    
    def mean(self, x: np.ndarray) -> float:
        work_group_size = 256
        n = x.size
        num_groups = (n + work_group_size - 1) // work_group_size
        temp_result = np.empty(num_groups, dtype=np.float32)
        
        x_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
        temp_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, temp_result.nbytes)
        
        self.reduction_program.mean_reduce(
            self.queue, (num_groups * work_group_size,), (work_group_size,),
            x_buf, temp_buf, cl.LocalMemory(np.float32(0).nbytes * work_group_size),
            np.int32(n)
        )
        
        cl.enqueue_copy(self.queue, temp_result, temp_buf)
        return float(temp_result.mean())
    
    def zeros(self, shape: Union[int, Tuple[int, ...]], dtype=np.float32) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        size = np.prod(shape)
        output = np.empty(shape, dtype=dtype)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.array_program.zeros_kernel(self.queue, (size,), None, out_buf)
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def ones(self, shape: Union[int, Tuple[int, ...]], dtype=np.float32) -> np.ndarray:
        if isinstance(shape, int):
            shape = (shape,)
        size = np.prod(shape)
        output = np.empty(shape, dtype=dtype)
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
        
        self.array_program.ones_kernel(self.queue, (size,), None, out_buf)
        cl.enqueue_copy(self.queue, output, out_buf)
        return output
    
    def zeros_like(self, x: np.ndarray) -> np.ndarray:
        return self.zeros(x.shape, dtype=x.dtype)
    
    def ones_like(self, x: np.ndarray) -> np.ndarray:
        return self.ones(x.shape, dtype=x.dtype)

opencl_ops = OpenCLOps() 