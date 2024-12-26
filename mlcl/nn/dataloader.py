import numpy as np
from typing import Tuple, Iterator, Optional
from ..core.tensor import Tensor
from ..core.opencl_utils import opencl_manager

class DataLoader:
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                 batch_size: int = 32, shuffle: bool = True):
        """
        Initialize DataLoader with data and parameters.
        
        Args:
            X: Input data
            y: Target data (optional)
            batch_size: Size of each batch
            shuffle: Whether to shuffle data each epoch
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)

        self.X_buffer = opencl_manager.allocate_buffer(X)
        if y is not None:
            self.y_buffer = opencl_manager.allocate_buffer(y)
        else:
            self.y_buffer = None
            
    def __iter__(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            X_batch = Tensor(self.X[batch_indices], buffer=self.X_buffer)
            
            if self.y is not None:
                y_batch = Tensor(self.y[batch_indices], buffer=self.y_buffer)
            else:
                y_batch = None
                
            yield X_batch, y_batch
            
    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size 