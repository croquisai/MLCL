# MLCL Documentation

## Core Components

#### `Tensor`
A fundamental class that represents n-dimensional arrays with automatic differentiation capabilities.
- Supports basic arithmetic operations (+, -, *, /, etc. (with broadcasting))
- Tracks gradients for backpropagation when `requires_grad=True`
- Implements broadcasting for operations between tensors of different shapes
- Provides shape manipulation methods like `reshape` and `view`

#### `MatMul`
A class that implements matrix multiplication using OpenCL acceleration.
- Optimized for GPU computation
- Supports automatic differentiation
- Handles gradient computation for both input matrices

### Neural Network Layers

#### `Linear`
A fully connected neural network layer implementing `y = xW + b`.
- Initializes weights using Xavier/Glorot initialization
- Supports automatic differentiation
- Parameters:
  - `in_features`: Number of input features
  - `out_features`: Number of output features

#### `Dropout`
A regularization layer that randomly zeros elements during training.
- Parameters:
  - `p`: Dropout probability (default: 0.5)
- Automatically scales outputs during training

#### `BatchNorm1d`
Implements 1D batch normalization for neural networks.
- Normalizes input features across the batch dimension
- Parameters:
  - `num_features`: Number of features to normalize
  - `eps`: Small constant for numerical stability (default: 1e-5)
  - `momentum`: Momentum for running statistics (default: 0.1)

#### `Conv2D`
2D convolutional layer for neural networks.
- Parameters:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `kernel_size`: Size of the convolving kernel
  - `stride`: Convolution stride (default: 1)
  - `padding`: Zero-padding size (default: 0)

### Activation Functions

#### `sigmoid`
Implements the sigmoid activation function: σ(x) = 1/(1 + e^(-x))
- Supports both Tensor and numpy array inputs
- Includes automatic differentiation

#### `relu`
Implements the Rectified Linear Unit (ReLU): f(x) = max(0, x)
- Accelerated implementation
- Supports automatic differentiation

#### `tanh`
Implements the hyperbolic tangent activation function
- Accelerated implementation
- Supports automatic differentiation

#### `leaky_relu`
Implements Leaky ReLU activation function: f(x) = x if x > 0 else αx
- Parameters:
  - `alpha`: Slope for negative values (default: 0.01)
- Accelerated implementation
- Supports automatic differentiation

#### `elu`
Implements Exponential Linear Unit: f(x) = x if x > 0 else α(e^x - 1)
- Parameters:
  - `alpha`: Scale for negative values (default: 1.0)
- Accelerated implementation
- Supports automatic differentiation

#### `selu`
Implements Scaled Exponential Linear Unit
- Self-normalizing properties
- Parameters:
  - `alpha`: Scale for negative values (default: 1.67326324)
  - `scale`: Scale factor (default: 1.05070098)
- Accelerated implementation
- Supports automatic differentiation

#### `softplus`
Implements Softplus activation: f(x) = ln(1 + e^x)
- Smooth approximation of ReLU
- Parameters:
  - `beta`: Smoothing factor (default: 1.0)
- Accelerated implementation
- Supports automatic differentiation