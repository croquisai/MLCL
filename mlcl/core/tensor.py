import numpy as np
import pyopencl as cl
from numpy import empty, float32
from .opencl_utils import opencl_manager

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None if not requires_grad else np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = None
        self.tensor_program = opencl_manager.get_program(program_name="tensor_ops")

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def transpose(self, *dims):
        """
        Transpose the tensor along the specified dimensions.
        If no dimensions are specified, reverses all dimensions.
        """
        if not dims:
            # If no dimensions specified, reverse all dimensions
            dims = tuple(range(len(self.shape)-1, -1, -1))
        
        # Create the transposed data
        new_data = np.transpose(self.data, dims)
        result = Tensor(new_data, requires_grad=self.requires_grad)
        
        if self.requires_grad and self.grad is not None:
            result.grad = np.transpose(self.grad, dims)
            
            def _backward(grad):
                # Reverse the transpose operation for gradient
                reverse_dims = [0] * len(dims)
                for i, d in enumerate(dims):
                    reverse_dims[d] = i
                self.backward(np.transpose(grad, reverse_dims))
            
            result._backward = _backward
        
        return result

    def reshape(self, *shape):
        new_data = self.data.reshape(*shape)
        result = Tensor(new_data, requires_grad=self.requires_grad)
        if self.requires_grad and self.grad is not None:
            result.grad = self.grad.reshape(*shape)
        return result

    def view(self, *shape):
        return self.reshape(*shape)

    def _check_shape_match(self, other):
        if isinstance(other, (int, float)):
            return True
        
        shape1 = self.shape
        shape2 = other.shape
        len1, len2 = len(shape1), len(shape2)
        
        if len1 < len2:
            shape1 = (1,) * (len2 - len1) + shape1
        elif len2 < len1:
            shape2 = (1,) * (len1 - len2) + shape2
            
        for a, b in zip(shape1, shape2):
            if a != 1 and b != 1 and a != b:
                return False
        return True

    def _get_broadcast_shape(self, other):
        if isinstance(other, (int, float)):
            return self.shape
            
        shape1 = self.shape
        shape2 = other.shape
        len1, len2 = len(shape1), len(shape2)
        
        if len1 < len2:
            shape1 = (1,) * (len2 - len1) + shape1
        elif len2 < len1:
            shape2 = (1,) * (len1 - len2) + shape2
            
        return tuple(max(a, b) for a, b in zip(shape1, shape2))

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        
        if grad is None:
            grad = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self._backward is not None:
            self._backward(grad)

    def __add__(self, other):
        if not self._check_shape_match(other):
            raise ValueError(f"Shapes {self.shape} and {other.shape} cannot be broadcast together")
        
        output_shape = self._get_broadcast_shape(other)
        result = empty(output_shape, dtype=float32)

        A_shape = np.array(self.shape if len(self.shape) >= len(output_shape) 
                          else (1,) * (len(output_shape) - len(self.shape)) + self.shape, dtype=np.int32)
        B_shape = np.array(other.shape if len(other.shape) >= len(output_shape)
                          else (1,) * (len(output_shape) - len(other.shape)) + other.shape, dtype=np.int32)
        C_shape = np.array(output_shape, dtype=np.int32)
        
        a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
        b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=other.data)
        c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
        
        a_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A_shape)
        b_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B_shape)
        c_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C_shape)

        total_size = result.size
        self.tensor_program.add_broadcast(opencl_manager.queue, (total_size,), None, 
                                        a_buf, b_buf, c_buf,
                                        a_shape_buf, b_shape_buf, c_shape_buf,
                                        np.int32(len(output_shape)))
        
        cl.enqueue_copy(opencl_manager.queue, result, c_buf)
        out = Tensor(result, requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        
        if out.requires_grad:
            out.saved_tensors = (self, other)
            out.saved_shapes = (A_shape, B_shape, C_shape)
            
            def _backward(grad):
                if self.requires_grad:
                    axes = tuple(i for i, (a, c) in enumerate(zip(A_shape, C_shape)) if a == 1 and c > 1)
                    grad_self = np.sum(grad, axis=axes, keepdims=True) if axes else grad
                    self.backward(grad_self)
                
                if isinstance(other, Tensor) and other.requires_grad:
                    axes = tuple(i for i, (b, c) in enumerate(zip(B_shape, C_shape)) if b == 1 and c > 1)
                    grad_other = np.sum(grad, axis=axes, keepdims=True) if axes else grad
                    other.backward(grad_other)
                    
            out._backward = _backward
            
        return out

    def __neg__(self):
        result = empty(self.data.shape, dtype=float32)
        
        a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
        b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
        
        self.tensor_program.neg(opencl_manager.queue, (self.data.size,), None, a_buf, b_buf)
        cl.enqueue_copy(opencl_manager.queue, result, b_buf)
        
        return Tensor(result, requires_grad=self.requires_grad)

    def __sub__(self, other):
        if not self._check_shape_match(other):
            raise ValueError(f"Shape mismatch for subtraction: {self.shape} vs {other.shape}")
            
        result = empty(self.data.shape, dtype=float32)
        
        a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
        b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=other.data)
        c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

        total_size = self.data.size
        self.tensor_program.subtract(opencl_manager.queue, (total_size,), None, a_buf, b_buf, c_buf)
        
        cl.enqueue_copy(opencl_manager.queue, result, c_buf)
        return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            result = empty(self.data.shape, dtype=float32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            self.tensor_program.div_scalar(opencl_manager.queue, (self.data.size,), None, a_buf, float32(other), c_buf)
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            
            return Tensor(result, requires_grad=self.requires_grad)
        else:
            if not self._check_shape_match(other):
                raise ValueError(f"Shape mismatch for division: {self.shape} vs {other.shape}")
            
            result = empty(self.data.shape, dtype=float32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=other.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            self.tensor_program.div(opencl_manager.queue, (self.data.size,), None, a_buf, b_buf, c_buf)
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            
            return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result = empty(self.data.shape, dtype=float32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            self.tensor_program.pow_scalar(opencl_manager.queue, (self.data.size,), None, a_buf, float32(other), c_buf)
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            
            return Tensor(result, requires_grad=self.requires_grad)
        else:
            if not self._check_shape_match(other):
                raise ValueError(f"Shape mismatch for power operation: {self.shape} vs {other.shape}")
            
            result = empty(self.data.shape, dtype=float32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=other.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            self.tensor_program.pow(opencl_manager.queue, (self.data.size,), None, a_buf, b_buf, c_buf)
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            
            return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            result = empty(self.data.shape, dtype=float32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            self.tensor_program.mul_scalar(opencl_manager.queue, (self.data.size,), None, a_buf, float32(other), c_buf)
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            
            return Tensor(result, requires_grad=self.requires_grad)
        else:
            if not self._check_shape_match(other):
                raise ValueError(f"Shape mismatch for multiplication: {self.shape} vs {other.shape}")
            
            output_shape = self._get_broadcast_shape(other)
            result = empty(output_shape, dtype=float32)
            
            A_shape = np.array(self.shape if len(self.shape) >= len(output_shape) 
                            else (1,) * (len(output_shape) - len(self.shape)) + self.shape, dtype=np.int32)
            B_shape = np.array(other.shape if len(other.shape) >= len(output_shape)
                            else (1,) * (len(output_shape) - len(other.shape)) + other.shape, dtype=np.int32)
            C_shape = np.array(output_shape, dtype=np.int32)
            
            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            b_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=other.data)
            c_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            
            a_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A_shape)
            b_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B_shape)
            c_shape_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=C_shape)

            total_size = result.size
            self.tensor_program.mul_broadcast(opencl_manager.queue, (total_size,), None, 
                                            a_buf, b_buf, c_buf,
                                            a_shape_buf, b_shape_buf, c_shape_buf,
                                            np.int32(len(output_shape)))
            
            cl.enqueue_copy(opencl_manager.queue, result, c_buf)
            return Tensor(result, requires_grad=self.requires_grad or other.requires_grad)

    def __rmul__(self, other):
        return self.__mul__(other)

    def mean(self, axis=None, dtype=None, out=None, **kwargs):
        if dtype is not None or out is not None:
            result = np.mean(self.data, axis=axis, dtype=dtype, out=out, **kwargs)
            return Tensor(result, requires_grad=self.requires_grad)
            
        if axis is None:
            program = opencl_manager.get_program(program_name="reduction_ops")
            
            work_group_size = 256
            n = self.data.size
            num_groups = (n + work_group_size - 1) // work_group_size
            temp_result = empty(num_groups, dtype=float32)

            a_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            temp_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, temp_result.nbytes)

            program.mean_reduce(
                opencl_manager.queue, (num_groups * work_group_size,), (work_group_size,),
                a_buf, temp_buf, cl.LocalMemory(float32(0).nbytes * work_group_size), np.int32(n)
            )
            cl.enqueue_copy(opencl_manager.queue, temp_result, temp_buf)

            mean_value = float32(temp_result.sum() / n)
            return Tensor([mean_value], requires_grad=self.requires_grad)
        else:
            if isinstance(axis, int):
                axis = (axis,)
            elif not isinstance(axis, tuple):
                axis = tuple(axis)
            
            axis = tuple(ax if ax >= 0 else len(self.shape) + ax for ax in axis)
            
            output_shape = [dim for i, dim in enumerate(self.shape) if i not in axis]
            if not output_shape:
                output_shape = (1,)
            
            reduction_size = 1
            for ax in axis:
                reduction_size *= self.shape[ax]
            
            program = opencl_manager.get_program(program_name="reduction_ops")
            
            input_shape = np.array(self.shape, dtype=np.int32)
            input_strides = np.array([np.prod(input_shape[i+1:], dtype=np.int32) for i in range(len(input_shape))] + [1], dtype=np.int32)
            output_shape = np.array(output_shape, dtype=np.int32)
            output_strides = np.array([np.prod(output_shape[i+1:], dtype=np.int32) for i in range(len(output_shape))], dtype=np.int32)
            
            result = empty(output_shape, dtype=float32)
            
            input_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
            output_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
            input_strides_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=input_strides)
            output_strides_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=output_strides)
            reduction_axes_buf = cl.Buffer(opencl_manager.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, 
                                        hostbuf=np.array(axis, dtype=np.int32))
            
            output_size = np.prod(output_shape)
            program.mean_reduce_axis(
                opencl_manager.queue, (output_size,), None,
                input_buf, output_buf,
                input_strides_buf, output_strides_buf,
                reduction_axes_buf,
                np.int32(len(axis)),
                np.int32(reduction_size),
                np.int32(output_size)
            )
            
            cl.enqueue_copy(opencl_manager.queue, result, output_buf)
            return Tensor(result, requires_grad=self.requires_grad)

    def softmax(self, dim=-1):
        if dim < 0:
            dim = len(self.shape) + dim

        broadcast_shape = list(self.shape)
        broadcast_shape[dim] = 1

        x_max = self.data.max(axis=dim, keepdims=True)
        exp_x = np.exp(self.data - x_max)

        softmax_output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        
        result = Tensor(softmax_output, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(grad):
                s = softmax_output
                grad_self = s * (grad - (s * grad).sum(axis=dim, keepdims=True))
                self.backward(grad_self)
                
            result._backward = _backward
            
        return result

    def masked_fill(self, mask, value):
        """
        Fills elements of self tensor with value where mask is True.
        """
        if not isinstance(mask, Tensor):
            mask = Tensor(mask)
            
        result = np.where(mask.data, value, self.data)
        return Tensor(result, requires_grad=self.requires_grad)