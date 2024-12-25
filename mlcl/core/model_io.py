import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Union, List
from .tensor import Tensor

def save_parameters(model_params: Union[Dict[str, Tensor], List[Tensor]], filepath: str) -> None:
    """
    Save model parameters to a binary file.
    
    Args:
        model_params: Dictionary of parameter name to Tensor, or list of Tensors
        filepath: Path where to save the parameters
    """
    if isinstance(model_params, dict):
        save_dict = {name: param.data for name, param in model_params.items()}
    else:
        save_dict = {'param_{}'.format(i): param.data for i, param in enumerate(model_params)}

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)

def load_parameters(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load model parameters from a binary file.
    
    Args:
        filepath: Path to the saved parameters file
        
    Returns:
        Dictionary mapping parameter names to numpy arrays
    """
    with open(filepath, 'rb') as f:
        params_dict = pickle.load(f)
    return params_dict

def apply_parameters(model_params: Union[Dict[str, Tensor], List[Tensor]], loaded_params: Dict[str, np.ndarray]) -> None:
    """
    Apply loaded parameters to model tensors.
    
    Args:
        model_params: Dictionary of parameter name to Tensor, or list of Tensors
        loaded_params: Dictionary of parameter name to numpy array
    """
    if isinstance(model_params, dict):
        for name, param in model_params.items():
            if name in loaded_params:
                param.data = loaded_params[name]
    else:
        for i, param in enumerate(model_params):
            key = 'param_{}'.format(i)
            if key in loaded_params:
                param.data = loaded_params[key]

class ModelIO:
    """Helper class for saving and loading model parameters."""
    
    def __init__(self, model_dir: str = "saved_models"):
        """
        Initialize ModelIO with a directory for saving models.
        
        Args:
            model_dir: Directory where models will be saved
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name: str, parameters: Union[Dict[str, Tensor], List[Tensor]]) -> None:
        """
        Save model parameters with a given name.
        
        Args:
            name: Name of the model
            parameters: Dictionary of parameter name to Tensor, or list of Tensors
        """
        filepath = self.model_dir / f"{name}.pkl"
        save_parameters(parameters, str(filepath))
    
    def load(self, name: str) -> Dict[str, np.ndarray]:
        """
        Load model parameters by name.
        
        Args:
            name: Name of the model to load
            
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        filepath = self.model_dir / f"{name}.pkl"
        return load_parameters(str(filepath))
    
    def apply(self, name: str, parameters: Union[Dict[str, Tensor], List[Tensor]]) -> None:
        """
        Load and apply parameters to model tensors.
        
        Args:
            name: Name of the model to load
            parameters: Dictionary of parameter name to Tensor, or list of Tensors
        """
        loaded_params = self.load(name)
        apply_parameters(parameters, loaded_params) 