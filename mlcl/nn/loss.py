import numpy as np
from typing import Union, Optional
import warnings
from ..core.tensor import Tensor
from ..core.accelerated_ops import accelerated_ops

class Loss:
    """Base class for all loss functions."""
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Backward pass to compute gradients.
        
        Args:
            y_pred: Predicted values
            y_true: Ground truth values
            
        Returns:
            np.ndarray: Gradients with respect to y_pred
        """
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error Loss."""
    
    def forward(self, y_pred, y_true):
        """
        Compute Mean Squared Error Loss: (1/n) * Σ(y_pred - y_true)²
        """
        if isinstance(y_pred, Tensor):
            self.y_pred = y_pred
            self.y_true = y_true
            diff = y_pred.data - y_true.data
            self.diff = diff
            loss_value = np.mean(diff * diff)
            loss_tensor = Tensor([loss_value], requires_grad=True)
            
            def _backward(grad=None):
                if grad is None:
                    grad = np.ones_like(loss_tensor.data)
                batch_size = np.prod(self.diff.shape)
                grad_factor = 2.0 * self.diff / batch_size
                if y_pred.requires_grad:
                    y_pred.backward(grad_factor)
                
            loss_tensor._backward = _backward
            return loss_tensor
            
        return np.mean((y_pred - y_true) * (y_pred - y_true))
    
    def backward(self, y_pred=None, y_true=None):
        """
        Compute gradient of MSE with respect to y_pred: 2 * (y_pred - y_true) / n
        """
        if y_pred is None:
            y_pred = self.y_pred
        if y_true is None:
            y_true = self.y_true
            
        if isinstance(y_pred, Tensor):
            batch_size = np.prod(self.diff.shape)
            grad = 2.0 * self.diff / batch_size
            if y_pred.requires_grad:
                y_pred.backward(grad)
            return grad
        return 2 * (y_pred - y_true) / y_pred.size


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss for multi-class classification."""
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small constant to avoid log(0)
        """
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Cross Entropy Loss: -Σ(y_true * log(y_pred))
        
        Args:
            y_pred: Predicted probabilities after softmax
            y_true: One-hot encoded ground truth labels
        """
        
        y_pred = accelerated_ops.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -accelerated_ops.sum(y_true * accelerated_ops.log(y_pred)) / y_pred.shape[0]
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Cross Entropy with respect to y_pred
        """

        y_pred = accelerated_ops.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -(y_true / y_pred) / y_pred.shape[0]


class BinaryCrossEntropyLoss(Loss):
    """Binary Cross Entropy Loss for binary classification."""
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small constant to avoid log(0)
        """
        self.epsilon = epsilon
    
    def forward(self, y_pred, y_true):
        """
        Compute Binary Cross Entropy Loss: -Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        """
        if isinstance(y_pred, Tensor):
            self.y_pred = y_pred
            self.y_true = y_true

            pred_data = accelerated_ops.clip(y_pred.data, self.epsilon, 1 - self.epsilon)
            true_data = y_true.data

            loss_value = -accelerated_ops.mean(
                true_data * accelerated_ops.log(pred_data) + 
                (1 - true_data) * accelerated_ops.log(1 - pred_data)
            )

            loss_tensor = Tensor([loss_value], requires_grad=True)
            
            def _backward(grad=None):
                if grad is None:
                    grad = accelerated_ops.ones_like(loss_tensor.data)

                pred_data_clip = accelerated_ops.clip(y_pred.data, self.epsilon, 1 - self.epsilon)
                grad_factor = -(true_data / pred_data_clip - 
                              (1 - true_data) / (1 - pred_data_clip)) / pred_data_clip.shape[0]
                
                y_pred.backward(grad_factor)
            
            loss_tensor._backward = _backward
            return loss_tensor

        y_pred = accelerated_ops.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -accelerated_ops.mean(y_true * accelerated_ops.log(y_pred) + (1 - y_true) * accelerated_ops.log(1 - y_pred))
    
    def backward(self, y_pred=None, y_true=None):
        """
        Compute gradient of Binary Cross Entropy with respect to y_pred
        """
        if y_pred is None:
            y_pred = self.y_pred
        if y_true is None:
            y_true = self.y_true
            
        if isinstance(y_pred, Tensor):
            pred_data = accelerated_ops.clip(y_pred.data, self.epsilon, 1 - self.epsilon)
            grad = -(y_true.data / pred_data - 
                    (1 - y_true.data) / (1 - pred_data)) / pred_data.shape[0]
            y_pred.backward(grad)
            return grad

        y_pred = accelerated_ops.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_pred.shape[0]


class MAELoss(Loss):
    """Mean Absolute Error Loss."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Mean Absolute Error Loss: (1/n) * Σ|y_pred - y_true|
        """
        return accelerated_ops.mean(accelerated_ops.abs(y_pred - y_true))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MAE with respect to y_pred: sign(y_pred - y_true) / n
        """
        return accelerated_ops.sign(y_pred - y_true) / y_pred.size
