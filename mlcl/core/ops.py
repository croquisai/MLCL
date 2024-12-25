import pyopencl as cl
from numpy import empty, int32, float32
from ..core.tensor import Tensor
from numpy import dot
from .opencl_utils import opencl_manager
import numpy as np

class MatMul:
    def __init__(self):
        self.program = opencl_manager.get_program(program_name="matmul")

    def forward(self, a, b):
        """
        Forward pass of matrix multiplication.
        Handles both 2D and batched inputs.
        
        For 2D inputs:
            a: (M, K), b: (K, N) -> output: (M, N)
        For batched inputs:
            a: (..., M, K), b: (..., K, N) -> output: (..., M, N)
        """
        a_shape = a.data.shape
        b_shape = b.data.shape

        if len(a_shape) > 2:
            batch_dims = a_shape[:-2]
            batch_size = np.prod(batch_dims)
            M, K = a_shape[-2:]
            _, N = b_shape[-2:]

            a_2d = a.data.reshape(batch_size, M, K)
            b_2d = b.data.reshape(batch_size, K, N)

            output = np.zeros((batch_size, M, N), dtype=np.float32)

            for i in range(batch_size):
                a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_2d[i])
                b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_2d[i])
                c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, M * N * 4)
                
                self.program.matmul(
                    opencl_manager.queue, (M, N), None,
                    a_buf, b_buf, c_buf,
                    int32(M), int32(N), int32(K)
                )
                
                cl.enqueue_copy(opencl_manager.queue, output[i], c_buf)

            output = output.reshape(*batch_dims, M, N)
        else:
            M, K = a.data.shape
            K2, N = b.data.shape
            
            if K != K2:
                raise ValueError(f"Incompatible shapes for matrix multiplication: {a.data.shape} and {b.data.shape}")
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a.data)
            b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.data)
            
            output = empty((M, N), dtype=float32)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, output.nbytes)
            
            self.program.matmul(
                opencl_manager.queue, (M, N), None,
                a_buf, b_buf, c_buf,
                int32(M), int32(N), int32(K)
            )
            
            cl.enqueue_copy(opencl_manager.queue, output, c_buf)
        
        out = Tensor(output, requires_grad=a.requires_grad or b.requires_grad)
        
        if out.requires_grad:
            self.a = a
            self.b = b
            self.batch_dims = batch_dims if len(a_shape) > 2 else None
            
            def _backward(grad):
                if len(a_shape) > 2:
                    grad_2d = grad.reshape(batch_size, M, N)
                    a_2d = a.data.reshape(batch_size, M, K)
                    b_2d = b.data.reshape(batch_size, K, N)
                    
                    if a.requires_grad:
                        a_grad = np.zeros((batch_size, M, K), dtype=np.float32)
                        for i in range(batch_size):
                            a_grad[i] = grad_2d[i] @ b_2d[i].T
                        a_grad = a_grad.reshape(*batch_dims, M, K)
                        a.backward(a_grad)
                    
                    if b.requires_grad:
                        b_grad = np.zeros((batch_size, K, N), dtype=np.float32)
                        for i in range(batch_size):
                            b_grad[i] = a_2d[i].T @ grad_2d[i]
                        b_grad = b_grad.reshape(*batch_dims, K, N)
                        b.backward(b_grad)
                else:
                    if a.requires_grad:
                        a_grad = grad @ b.data.T
                        a.backward(a_grad)
                    
                    if b.requires_grad:
                        b_grad = a.data.T @ grad
                        b.backward(b_grad)
            
            out._backward = _backward
        
        return out