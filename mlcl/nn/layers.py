from ..core.ops import MatMul
from ..core.tensor import Tensor
from numpy import random, zeros
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # Xavier/Glorot (oh dear...)
        weight_scale = np.sqrt(2.0 / (in_features + out_features))
        self.weights = Tensor(random.randn(in_features, out_features) * weight_scale, requires_grad=True)
        self.bias = Tensor(zeros(out_features), requires_grad=True)
        self.matmul = MatMul()
    
    def forward(self, x):
        self.input = x

        out = self.matmul.forward(x, self.weights)

        result = out + self.bias

        self.output_before_bias = out
        
        def _backward(grad):
            if self.bias.requires_grad:
                self.bias.grad = grad.sum(axis=0)

            if self.weights.requires_grad:
                self.weights.grad = self.input.data.T @ grad
            
            if self.input.requires_grad:
                input_grad = grad @ self.weights.data.T
                self.input.backward(input_grad)
        
        result._backward = _backward
        return result

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def forward(self, x):
        if not self.training:
            return x

        self.mask = (random.rand(*x.data.shape) > self.p) / (1 - self.p)
        result = x * Tensor(self.mask)
        
        def _backward(grad):
            if x.requires_grad:
                x.backward(grad * self.mask)
        
        result._backward = _backward
        return result

class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        self.training = True
    
    def forward(self, x):
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0) + self.eps

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_norm = (x.data - mean) / np.sqrt(var)
        else:
            x_norm = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        out = self.gamma.data * x_norm + self.beta.data
        result = Tensor(out, requires_grad=x.requires_grad)
        
        def _backward(grad):
            if x.requires_grad:
                N = x.data.shape[0]
                x_mu = x.data - mean
                std_inv = 1 / np.sqrt(var)

                dx_norm = grad * self.gamma.data
                dvar = -0.5 * np.sum(dx_norm * x_mu, axis=0) * std_inv**3
                dmean = -np.sum(dx_norm * std_inv, axis=0) - 2 * dvar * np.mean(x_mu, axis=0)
                dx = dx_norm * std_inv + 2 * dvar * x_mu / N + dmean / N

                if self.gamma.requires_grad:
                    self.gamma.grad = np.sum(grad * x_norm, axis=0)
                if self.beta.requires_grad:
                    self.beta.grad = np.sum(grad, axis=0)
                
                x.backward(dx)
        
        result._backward = _backward
        return result

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        scale = np.sqrt(2.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1] + out_channels))
        kernel_shape = (out_channels, in_channels, *self.kernel_size)
        self.kernels = Tensor(random.randn(*kernel_shape) * scale, requires_grad=True)
        self.bias = Tensor(zeros(out_channels), requires_grad=True)
    
    def _pad(self, x):
        if self.padding == (0, 0):
            return x
        return np.pad(x, ((0,0), (0,0), (self.padding[0],self.padding[0]), 
                         (self.padding[1],self.padding[1])), mode='constant')
    
    def forward(self, x):
        N, C, H, W = x.data.shape
        self.input_shape = x.data.shape

        x_padded = self._pad(x.data)

        H_out = (H + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = np.zeros((N, self.out_channels, H_out, W_out))

        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride[0]
                        w_start = w * self.stride[1]
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        receptive_field = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, c_out, h, w] = np.sum(receptive_field * self.kernels.data[c_out]) + self.bias.data[c_out]
        
        result = Tensor(out, requires_grad=x.requires_grad)
        
        def _backward(grad):
            if x.requires_grad:
                dx_padded = np.zeros_like(x_padded)
            
            if self.kernels.requires_grad:
                self.kernels.grad = np.zeros_like(self.kernels.data)
            
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis=(0,2,3))

            for n in range(N):
                for c_out in range(self.out_channels):
                    for h in range(H_out):
                        for w in range(W_out):
                            h_start = h * self.stride[0]
                            w_start = w * self.stride[1]
                            h_end = h_start + self.kernel_size[0]
                            w_end = w_start + self.kernel_size[1]
                            
                            if x.requires_grad:
                                dx_padded[n, :, h_start:h_end, w_start:w_end] += \
                                    self.kernels.data[c_out] * grad[n, c_out, h, w]
                            
                            if self.kernels.requires_grad:
                                self.kernels.grad[c_out] += \
                                    x_padded[n, :, h_start:h_end, w_start:w_end] * grad[n, c_out, h, w]
            
            if x.requires_grad:
                if self.padding[0] > 0 or self.padding[1] > 0:
                    dx = dx_padded[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
                else:
                    dx = dx_padded
                x.backward(dx)
        
        result._backward = _backward
        return result