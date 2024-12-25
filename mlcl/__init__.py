from .core.opencl_utils import opencl_manager, OpenCLManager
from .core.tensor import Tensor
from .core.ops import MatMul
from .nn.layers import Linear, Conv2D, BatchNorm1D
from .nn.activations import sigmoid, relu, tanh
from .nn.loss import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, MAELoss, Loss
from .nn.optimizers import Optimizer, SGD
from .core.model_io import ModelIO
from .hdl.pipeline import ModelToHDLPipeline

__all__ = ['Tensor', 'MatMul', 'opencl_manager', 'ops_manager','OpenCLManager']

__nn__ = ['Tensor', 'MatMul', 'Linear', 'Conv2D', 'BatchNorm1D', 'sigmoid', 'MSELoss', 'CrossEntropyLoss', 
           'BinaryCrossEntropyLoss', 'MAELoss', 'Loss', 'Optimizer', 'SGD',
           'opencl_manager', 'ops_manager', 'relu', 'tanh', 'OpenCLManager', 'ModelIO', 'ModelToHDLPipeline']

__activations__ = ['sigmoid', 'relu', 'tanh']

__loss__ = ['MSELoss', 'CrossEntropyLoss', 'BinaryCrossEntropyLoss', 'MAELoss', 'Loss']

__optimizers__ = ['Optimizer', 'SGD']

__opencl__ = ['opencl_manager', 'ops_manager', 'OpenCLManager']

__hdl__ = ['ModelToHDLPipeline']