import pyopencl as cl
from numpy import empty, int32, float32
from ..core.tensor import Tensor
from numpy import dot
from .opencl_utils import opencl_manager

class MatMul:
    def __init__(self):
        self.program = opencl_manager.get_program(program_name="matmul")

    def forward(self, a, b):
        M, K = a.data.shape
        K, N = b.data.shape

        a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a.data)
        b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b.data)
        
        c_np = empty((M, N), dtype=float32)
        c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, c_np.nbytes)
        
        self.program.matmul(
            opencl_manager.queue, (M, N), None,
            a_buf, b_buf, c_buf,
            int32(M), int32(N), int32(K)
        )
        
        cl.enqueue_copy(opencl_manager.queue, c_np, c_buf)
        out = Tensor(c_np, requires_grad=a.requires_grad or b.requires_grad)
        
        if out.requires_grad:
            self.a = a
            self.b = b
            
            def _backward(grad):
                if a.requires_grad:
                    a_grad = dot(grad, b.data.T)
                    a.backward(a_grad)

                if b.requires_grad:
                    b_grad = dot(a.data.T, grad)
                    b.backward(b_grad)
                    
            out._backward = _backward
            
        return out