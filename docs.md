# MLCL Documentation

## Core Components

#### `Tensor`
A fundamental class that represents n-dimensional arrays with automatic differentiation capabilities.
- Supports basic arithmetic operations (+, -, *, /, etc.)
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

### Loss Functions

#### `MSELoss`
Mean Squared Error loss function: L = (1/n)Σ(y_pred - y_true)²
- Suitable for regression problems
- Computes gradients for backpropagation

#### `CrossEntropyLoss`
Cross-entropy loss for multi-class classification
- Expects input as probabilities (after softmax)
- Handles numerical stability with epsilon parameter
- Parameters:
  - `epsilon`: Small constant to avoid log(0) (default: 1e-15)

#### `BinaryCrossEntropyLoss`
Binary cross-entropy loss for binary classification
- Suitable for binary classification problems
- Handles numerical stability
- Parameters:
  - `epsilon`: Small constant to avoid log(0) (default: 1e-15)

#### `MAELoss`
Mean Absolute Error loss: L = (1/n)Σ|y_pred - y_true|
- Alternative to MSE for regression
- Less sensitive to outliers

### Optimizers

#### `Optimizer`
Base class for all optimizers
- Provides common functionality for parameter updates
- Implements gradient zeroing

#### `SGD`
Stochastic Gradient Descent optimizer with momentum
- Parameters:
  - `parameters`: List of parameters to optimize
  - `learning_rate`: Learning rate (default: 0.01)
  - `momentum`: Momentum factor (default: 0.9)
  - `clip_value`: Gradient clipping threshold (default: 1.0)
- Features:
  - Momentum for faster convergence
  - Gradient clipping to prevent exploding gradients