# MLCL

MLCL is a OpenCL-focused machine learning library with the following design goals:
- Small install size
- Easy to use
- Fast
- Flexible
- OpenCL-based for GPU acceleration
- JIT for CPUs (if OpenCL isn't supported!)

| Import Type | Imported Item(s) | Description |
|-----------------------|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Core Utilities | opencl_manager, OpenCLManager | Manages OpenCL context and operations for GPU acceleration. |
| Core Tensor | Tensor | Represents n-dimensional arrays with automatic differentiation capabilities. |
| Core Operations | MatMul | Implements matrix multiplication using OpenCL acceleration. |
| Neural Network Layers | Linear | A fully connected layer in a neural network. |
| Neural Network Activations | sigmoid, relu, tanh | Activation functions used in neural networks to introduce non-linearity. |
| Neural Network Losses | MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, MAELoss, Loss | Various loss functions for training neural networks, measuring the difference between predicted and true values. |
| Neural Network Optimizers | Optimizer, SGD | Base class for optimizers and specific implementation of Stochastic Gradient Descent for parameter updates. |
| Public API | __all__ | Specifies the public objects that can be imported from the MLCL module. |
| Neural Network API | __nn__ | Lists neural network-related classes and functions available for import. |
| Activation Functions | __activations__ | Lists all activation functions available within MLCL. |
| Loss Functions | __loss__ | Lists all loss functions available within MLCL. |
| Optimizers | __optimizers__ | Lists all optimizer classes available within MLCL. |
| OpenCL Utilities | __opencl__ | Lists OpenCL-related objects available within MLCL. |

## Installation

For now, you will need to pull the repository and run `pip install .` from the root directory.

```bash
git clone https://github.com/rndmcoolawsmgrbg/MLCL.git
cd MLCL
pip install .
```

