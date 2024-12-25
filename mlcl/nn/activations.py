import numpy as np
from ..core.tensor import Tensor
from ..core.accelerated_ops import accelerated_ops

def sigmoid(x):
    if isinstance(x, Tensor):
        result = 1 / (1 + accelerated_ops.exp(-x.data))
        out = Tensor(result, requires_grad=x.requires_grad)

        def _backward(grad):
            dx = grad * result * (1 - result)
            x.backward(dx)
            
        out._backward = _backward
        return out
    return 1 / (1 + accelerated_ops.exp(-x))

def sigmoid_backward(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    if isinstance(x, Tensor):
        result = accelerated_ops.relu(x.data)
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * (x.data > 0)
            x.backward(dx)
            
        out._backward = _backward
        return out
    return accelerated_ops.relu(x)

def tanh(x):
    if isinstance(x, Tensor):
        result = accelerated_ops.tanh(x.data)
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * (1 - result * result)
            x.backward(dx)
            
        out._backward = _backward
        return out
    return accelerated_ops.tanh(x)