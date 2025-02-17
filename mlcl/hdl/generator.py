from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import datetime
import os

message = f"""// HDL generated by mlcl.hdl.generator.HDLGenerator
// Author: {os.getlogin()} on {os.name}
// Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

class HDLGenerator(ABC):
    """Base class for HDL generation."""
    
    def __init__(self, module_name: str, input_width: int = 32, output_width: int = 32):
        """
        Initialize HDL generator.
        
        Args:
            module_name: Name of the Verilog module
            input_width: Bit width of input values
            output_width: Bit width of output values
        """
        self.module_name = module_name
        self.input_width = input_width
        self.output_width = output_width
        self.parameters: Dict[str, Union[int, str]] = {}
        self.ports: Dict[str, Dict[str, str]] = {}
        self.internal_signals: List[Dict[str, str]] = []
        self.assignments: List[str] = []
        self.always_blocks: List[str] = []
        
    def add_parameter(self, name: str, value: Union[int, str]) -> None:
        """Add a parameter to the module."""
        self.parameters[name] = value
        
    def add_port(self, name: str, direction: str, width: Optional[Union[int, str]] = None) -> None:
        """
        Add an input or output port to the module.
        
        Args:
            name: Port name
            direction: Port direction ('input' or 'output')
            width: Port width. Can be an integer for fixed width or a string for parameterized width
        """
        if direction not in ['input', 'output']:
            raise ValueError(f"Invalid port direction: {direction}")
        
        port_def = {'direction': direction}
        if width is not None:
            port_def['width'] = width
            
        self.ports[name] = port_def
        
    def add_internal_signal(self, name: str, width: Optional[Union[int, str]] = None) -> None:
        """
        Add an internal signal (wire/reg) to the module.
        
        Args:
            name: Signal name
            width: Signal width. Can be an integer for fixed width or a string for parameterized width
        """
        signal_def = {'name': name}
        if width is not None:
            signal_def['width'] = width
            
        self.internal_signals.append(signal_def)
        
    def add_assignment(self, target: str, expression: str) -> None:
        """Add a continuous assignment."""
        self.assignments.append(f"assign {target} = {expression};")
        
    def add_always_block(self, sensitivity_list: List[str], statements: List[str]) -> None:
        """Add an always block with the given sensitivity list and statements."""
        sensitivity = " or ".join(sensitivity_list)
        block = f"always @({sensitivity}) begin\n"
        block += "\n".join(f"    {stmt}" for stmt in statements)
        block += "\nend"
        self.always_blocks.append(block)
        
    @abstractmethod
    def generate_logic(self) -> None:
        """Generate the module-specific logic. Must be implemented by subclasses."""
        pass
        
    def _format_width(self, width: Union[int, str]) -> str:
        """Format port/signal width for HDL generation."""
        if isinstance(width, int):
            return f"[{width-1}:0]"
        else:
            return f"[{width}-1:0]"
        
    def generate_hdl(self) -> str:
        self.generate_logic()
        
        parts = []
        
        if self.parameters:
            param_parts = [f"parameter {name} = {value}" for name, value in self.parameters.items()]
            parts.append(f"#(\n    {','.join(param_parts)}\n)")
        
        port_parts = []
        for name, port in self.ports.items():
            port_str = [port['direction']]
            if 'width' in port:
                port_str.append(self._format_width(port['width']))
            port_str.append(name)
            port_parts.append(" ".join(port_str))
        
        parts.extend([
            f"module {self.module_name}",
            f"({', '.join(port_parts)});",
            *[f"wire {self._format_width(s['width']) if 'width' in s else ''} {s['name']};" 
              for s in self.internal_signals],
            *self.assignments,
            *self.always_blocks,
            "endmodule"
        ])
        
        return "\n".join(parts)
        
    def write_to_file(self, filename: str) -> None:
        """Write the generated HDL to a file."""
        with open(filename, 'w') as f:
            f.write(message+self.generate_hdl()) 