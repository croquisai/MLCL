from ..core.ops import MatMul
from ..core.tensor import Tensor
from numpy import random, zeros
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        weight_scale = np.sqrt(1.0 / in_features)
        self.weights = Tensor(random.randn(in_features, out_features) * weight_scale, requires_grad=True)
        self.bias = Tensor(zeros(out_features), requires_grad=True)
        self.matmul = MatMul()
    
    def forward(self, x):
        """
        Forward pass of linear layer.
        Handles both 2D inputs (batch_size, in_features) and 
        3D inputs (batch_size, seq_len, in_features).
        """
        self.input = x
        batch_size = x.shape[0]

        x_reshaped = x.data.reshape(-1, x.shape[-1])

        out = x_reshaped @ self.weights.data + self.bias.data

        result = Tensor(out.reshape(batch_size, -1), requires_grad=x.requires_grad)
        
        def _backward(grad):
            grad = np.clip(grad, -1e3, 1e3)
            
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis=0)
                self.bias.grad = np.clip(self.bias.grad, -1e3, 1e3)

            if self.weights.requires_grad:
                self.weights.grad = x_reshaped.T @ grad
                # Clip weight gradients
                self.weights.grad = np.clip(self.weights.grad, -1e3, 1e3)
            
            if x.requires_grad:
                input_grad = grad @ self.weights.data.T
                input_grad = np.clip(input_grad, -1e3, 1e3)
                x.backward(input_grad)
        
        result._backward = _backward
        return result
    
    def parameters(self):
        return [self.weights, self.bias]

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
    
    def parameters(self):
        return []

class BatchNorm1D:
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
        input_shape = x.data.shape
        if len(input_shape) == 3:
            x_2d = x.data.reshape(-1, self.num_features)
        else:
            x_2d = x.data

        if self.training:
            mean = x_2d.mean(axis=0)
            var = x_2d.var(axis=0) + self.eps

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            x_norm = (x_2d - mean) / np.sqrt(var)
        else:
            x_norm = (x_2d - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        out = self.gamma.data * x_norm + self.beta.data

        if len(input_shape) == 3:
            out = out.reshape(*input_shape)
        
        result = Tensor(out, requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(grad):
                if len(input_shape) == 3:
                    grad = grad.reshape(-1, self.num_features)
                
                N = grad.shape[0]
                x_mu = x_2d - mean
                std_inv = 1 / np.sqrt(var)

                dx_norm = grad * self.gamma.data
                dvar = -0.5 * np.sum(dx_norm * x_mu, axis=0) * std_inv**3
                dmean = -np.sum(dx_norm * std_inv, axis=0) - 2 * dvar * np.mean(x_mu, axis=0)
                dx = dx_norm * std_inv + 2 * dvar * x_mu / N + dmean / N

                if len(input_shape) == 3:
                    dx = dx.reshape(*input_shape)

                if self.gamma.requires_grad:
                    self.gamma.grad = np.sum(grad * x_norm, axis=0)
                if self.beta.requires_grad:
                    self.beta.grad = np.sum(grad, axis=0)
                
                x.backward(dx)
        
        result._backward = _backward
        return result

    def parameters(self):
        return [self.gamma, self.beta]

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
    
    def parameters(self):
        return [self.kernels, self.bias]
    
    def _im2col(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = (H + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (W + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        x_padded = self._pad(x)
        col = np.zeros((N, C, self.kernel_size[0], self.kernel_size[1], out_h, out_w))
        
        for y in range(self.kernel_size[0]):
            y_max = y + self.stride[0] * out_h
            for x in range(self.kernel_size[1]):
                x_max = x + self.stride[1] * out_w
                col[:, :, y, x, :, :] = x_padded[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col

    def forward(self, x):
        N, C, H, W = x.data.shape
        self.input_shape = x.data.shape
        
        out_h = (H + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (W + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        col = self._im2col(x.data)
        kernel_col = self.kernels.data.reshape(self.out_channels, -1)
        
        out = kernel_col @ col.T
        if self.bias is not None:
            out += self.bias.data.reshape(-1, 1)
        
        out = out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
        result = Tensor(out, requires_grad=x.requires_grad)
        
        def _backward(grad):
            if x.requires_grad:
                dx_padded = np.zeros_like(x.data)
            
            if self.kernels.requires_grad:
                self.kernels.grad = np.zeros_like(self.kernels.data)
            
            if self.bias.requires_grad:
                self.bias.grad = np.sum(grad, axis=(0,2,3))

            for n in range(N):
                for c_out in range(self.out_channels):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride[0]
                            w_start = w * self.stride[1]
                            h_end = h_start + self.kernel_size[0]
                            w_end = w_start + self.kernel_size[1]
                            
                            if x.requires_grad:
                                dx_padded[n, :, h_start:h_end, w_start:w_end] += \
                                    self.kernels.data[c_out] * grad[c_out, h, w, n]
                            
                            if self.kernels.requires_grad:
                                self.kernels.grad[c_out] += \
                                    col[n * out_h * out_w + h * out_w + w] * grad[c_out, h, w, n]
            
            if x.requires_grad:
                if self.padding[0] > 0 or self.padding[1] > 0:
                    dx = dx_padded[:, :, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]]
                else:
                    dx = dx_padded
                x.backward(dx)
        
        result._backward = _backward
        return result
    
    def parameters(self):
        return [self.kernels, self.bias]
