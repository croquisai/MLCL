import numpy as np
from ..core.tensor import Tensor
from typing import List, Dict, Union, Tuple

class Optimizer:
    def __init__(self, parameters: List[Tensor]):
        self.parameters = parameters
        
    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
                
    def step(self):
        """Update parameters using their gradients."""
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, 
                 momentum: float = 0.9, clip_value: float = 1.0):
        super().__init__(parameters)
        self.learning_rate = lr
        self.momentum = momentum
        self.clip_value = clip_value
        self.momentum_buffers = {}

        if momentum > 0:
            for i, param in enumerate(parameters):
                self.momentum_buffers[id(param)] = np.zeros_like(param.data)
    
    def step(self):
        """Update parameters using SGD with optional momentum and gradient clipping."""
        for param in self.parameters:
            if param.grad is None:
                continue

            grad = param.grad
            if self.clip_value is not None:
                grad = np.clip(grad, -self.clip_value, self.clip_value)

            if self.momentum > 0:
                buffer = self.momentum_buffers[id(param)]
                if buffer.shape != param.data.shape:
                    buffer = np.zeros_like(param.data)
                    self.momentum_buffers[id(param)] = buffer
                buffer = self.momentum * buffer - self.learning_rate * grad
                self.momentum_buffers[id(param)] = buffer
                param.data = param.data + buffer
            else:
                param.data -= self.learning_rate * grad

            param.grad = None 