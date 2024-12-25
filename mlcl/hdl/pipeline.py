import os
from typing import List, Dict, Type, Optional
from ..nn.layers import Linear, Conv2D
from .layers import LinearHDL, Conv2DHDL
from .generator import HDLGenerator

class ModelToHDLPipeline:
    """Pipeline for converting neural network models to HDL."""
    
    LAYER_TO_GENERATOR = {
        Linear: LinearHDL,
        Conv2D: Conv2DHDL
    }
    
    def __init__(self, output_dir: str = "hdl_output", input_width: int = 32, 
                 output_width: int = 32):
        """
        Initialize the model-to-HDL pipeline.
        
        Args:
            output_dir: Directory to store generated HDL files
            input_width: Bit width for input values
            output_width: Bit width for output values
        """
        self.output_dir = output_dir
        self.input_width = input_width
        self.output_width = output_width
        os.makedirs(output_dir, exist_ok=True)
        
    def convert_layer(self, layer: object, name: Optional[str] = None) -> HDLGenerator:
        """
        Convert a single layer to HDL.
        
        Args:
            layer: Neural network layer instance
            name: Optional name for the generated module
            
        Returns:
            HDLGenerator instance for the layer
        """
        layer_type = type(layer)
        if layer_type not in self.LAYER_TO_GENERATOR:
            raise ValueError(f"Unsupported layer type: {layer_type}")
            
        generator_class = self.LAYER_TO_GENERATOR[layer_type]
        generator = generator_class(layer, self.input_width, self.output_width)
        
        if name:
            generator.module_name = name
            
        return generator
    
    def convert_model(self, model: List[object], model_name: str = "neural_network") -> List[str]:
        """
        Convert an entire model to HDL.
        
        Args:
            model: List of layer instances
            model_name: Name for the top-level module
            
        Returns:
            List of generated HDL file paths
        """
        
        hdl_files = []
        layer_modules = []
        
        for i, layer in enumerate(model):
            layer_name = f"{model_name}_layer_{i}"
            generator = self.convert_layer(layer, layer_name)

            hdl_path = os.path.join(self.output_dir, f"{layer_name}.v")
            generator.write_to_file(hdl_path)
            hdl_files.append(hdl_path)
            layer_modules.append(generator)

        top_module = self._generate_top_module(model_name, layer_modules)
        top_path = os.path.join(self.output_dir, f"{model_name}_top.v")
        with open(top_path, 'w') as f:
            f.write(top_module)
        hdl_files.append(top_path)
        
        return hdl_files
    
    def _generate_top_module(self, model_name: str, layer_modules: List[HDLGenerator]) -> str:
        """Generate top-level module that connects all layers."""
        hdl = []

        hdl.append(f"module {model_name}_top (")
        hdl.append("    input wire clk,")
        hdl.append("    input wire rst,")
        hdl.append("    input wire input_valid,")
        hdl.append(f"    input wire [{self.input_width}-1:0] input_data,")
        hdl.append("    output wire output_valid,")
        hdl.append(f"    output wire [{self.output_width}-1:0] output_data")
        hdl.append(");")

        for i in range(len(layer_modules) - 1):
            hdl.append(f"wire layer_{i}_valid;")
            hdl.append(f"wire [{self.output_width}-1:0] layer_{i}_data;")

        for i, generator in enumerate(layer_modules):
            hdl.append(f"\n// Layer {i}")
            hdl.append(f"{generator.module_name} layer_{i} (")
            hdl.append("    .clk(clk),")
            hdl.append("    .rst(rst),")

            if i == 0:
                hdl.append("    .input_valid(input_valid),")
                hdl.append("    .input_data(input_data),")
            else:
                hdl.append(f"    .input_valid(layer_{i-1}_valid),")
                hdl.append(f"    .input_data(layer_{i-1}_data),")

            if i == len(layer_modules) - 1:
                hdl.append("    .output_valid(output_valid),")
                hdl.append("    .output_data(output_data)")
            else:
                hdl.append(f"    .output_valid(layer_{i}_valid),")
                hdl.append(f"    .output_data(layer_{i}_data)")
            
            hdl.append(");")
            
        hdl.append("\nendmodule")
        return "\n".join(hdl) 