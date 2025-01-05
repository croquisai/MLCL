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

def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    f(x) = x if x > 0 else alpha * x
    """
    if isinstance(x, Tensor):
        result = np.where(x.data > 0, x.data, alpha * x.data)
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * np.where(x.data > 0, 1.0, alpha)
            x.backward(dx)
            
        out._backward = _backward
        return out
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """
    Exponential Linear Unit activation function.
    f(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    if isinstance(x, Tensor):
        pos_mask = x.data > 0
        result = np.where(pos_mask, x.data, 
                         alpha * (accelerated_ops.exp(x.data) - 1))
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * np.where(pos_mask, 1.0, 
                               alpha * accelerated_ops.exp(x.data))
            x.backward(dx)
            
        out._backward = _backward
        return out
    pos_mask = x > 0
    return np.where(pos_mask, x, alpha * (accelerated_ops.exp(x) - 1))

def selu(x, alpha=1.67326324, scale=1.05070098):
    """
    Scaled Exponential Linear Unit activation function.
    f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    """
    if isinstance(x, Tensor):
        pos_mask = x.data > 0
        result = scale * np.where(pos_mask, x.data,
                                alpha * (accelerated_ops.exp(x.data) - 1))
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * scale * np.where(pos_mask, 1.0,
                                       alpha * accelerated_ops.exp(x.data))
            x.backward(dx)
            
        out._backward = _backward
        return out
    pos_mask = x > 0
    return scale * np.where(pos_mask, x, alpha * (accelerated_ops.exp(x) - 1))

def softplus(x, beta=1.0):
    """
    Softplus activation function.
    f(x) = (1/beta) * log(1 + exp(beta * x))
    """
    if isinstance(x, Tensor):
        result = (1.0/beta) * accelerated_ops.log(1 + accelerated_ops.exp(beta * x.data))
        out = Tensor(result, requires_grad=x.requires_grad)
        
        def _backward(grad):
            dx = grad * (1.0/(1 + accelerated_ops.exp(-beta * x.data)))
            x.backward(dx)
            
        out._backward = _backward
        return out
    return (1.0/beta) * accelerated_ops.log(1 + accelerated_ops.exp(beta * x))