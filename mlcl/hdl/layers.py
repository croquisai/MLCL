import numpy as np
from typing import List, Optional, Tuple
from .generator import HDLGenerator
from ..nn.layers import Linear, Conv2D

class LinearHDL(HDLGenerator):
    """HDL generator for fully connected (Linear) layer."""
    
    def __init__(self, layer: Linear, input_width: int = 32, output_width: int = 32,
                 use_pipeline: bool = True):
        """
        Initialize LinearHDL generator.
        
        Args:
            layer: Linear layer instance
            input_width: Bit width of input values
            output_width: Bit width of output values
            use_pipeline: Whether to use pipelined architecture
        """
        super().__init__(f"linear_{layer.weights.shape[0]}x{layer.weights.shape[1]}", 
                        input_width, output_width)
        self.layer = layer
        self.use_pipeline = use_pipeline
        self.weights = layer.weights.data
        self.bias = layer.bias.data if layer.bias is not None else None
        
    def generate_logic(self) -> None:
        """Generate HDL logic for Linear layer."""
        in_features, out_features = self.weights.shape

        self.add_parameter("IN_FEATURES", in_features)
        self.add_parameter("OUT_FEATURES", out_features)
        self.add_parameter("DATA_WIDTH", self.input_width)

        self.add_port("clk", "input")
        self.add_port("rst", "input")
        self.add_port("input_valid", "input")
        self.add_port("input_data", "input", width=f"DATA_WIDTH*IN_FEATURES")
        self.add_port("output_valid", "output")
        self.add_port("output_data", "output", width=f"DATA_WIDTH*OUT_FEATURES")
        
        if self.use_pipeline:
            self.add_internal_signal("pipeline_valid", width=3)
            self.add_internal_signal("accumulator", width=f"DATA_WIDTH*OUT_FEATURES")
            
        for i in range(out_features):
            weights_str = ", ".join([f"{w:.6f}" for w in self.weights[:, i]])
            self.add_parameter(f"WEIGHTS_{i}", f"'{{{weights_str}}}")
            if self.bias is not None:
                self.add_parameter(f"BIAS_{i}", f"{self.bias[i]:.6f}")
                
        if self.use_pipeline:
            input_reg = [
                "if (rst) begin",
                "    pipeline_valid[0] <= 0;",
                "end else begin",
                "    pipeline_valid[0] <= input_valid;",
                "end"
            ]
            self.add_always_block(["posedge clk"], input_reg)

            mult_statements = []
            for i in range(out_features):
                mult_str = []
                for j in range(in_features):
                    mult_str.append(f"input_data[{j}*DATA_WIDTH +: DATA_WIDTH] * WEIGHTS_{i}[{j}]")
                mult_statements.append(f"accumulator[{i}*DATA_WIDTH +: DATA_WIDTH] <= {' + '.join(mult_str)};")
            
            compute = [
                "if (rst) begin",
                "    pipeline_valid[1] <= 0;",
                "end else begin",
                "    pipeline_valid[1] <= pipeline_valid[0];",
                *mult_statements,
                "end"
            ]
            self.add_always_block(["posedge clk"], compute)

            output_statements = []
            for i in range(out_features):
                if self.bias is not None:
                    output_statements.append(
                        f"output_data[{i}*DATA_WIDTH +: DATA_WIDTH] <= "
                        f"accumulator[{i}*DATA_WIDTH +: DATA_WIDTH] + BIAS_{i};"
                    )
                else:
                    output_statements.append(
                        f"output_data[{i}*DATA_WIDTH +: DATA_WIDTH] <= "
                        f"accumulator[{i}*DATA_WIDTH +: DATA_WIDTH];"
                    )
                    
            output = [
                "if (rst) begin",
                "    pipeline_valid[2] <= 0;",
                "    output_valid <= 0;",
                "end else begin",
                "    pipeline_valid[2] <= pipeline_valid[1];",
                "    output_valid <= pipeline_valid[2];",
                *output_statements,
                "end"
            ]
            self.add_always_block(["posedge clk"], output)
        else:
            output_statements = []
            for i in range(out_features):
                mult_str = []
                for j in range(in_features):
                    mult_str.append(f"input_data[{j}*DATA_WIDTH +: DATA_WIDTH] * WEIGHTS_{i}[{j}]")
                if self.bias is not None:
                    output_statements.append(
                        f"assign output_data[{i}*DATA_WIDTH +: DATA_WIDTH] = "
                        f"{' + '.join(mult_str)} + BIAS_{i};"
                    )
                else:
                    output_statements.append(
                        f"assign output_data[{i}*DATA_WIDTH +: DATA_WIDTH] = "
                        f"{' + '.join(mult_str)};"
                    )
            
            for stmt in output_statements:
                self.add_assignment(stmt.split("=")[0].strip(), stmt.split("=")[1].strip()[:-1])
                
            self.add_assignment("output_valid", "input_valid")


class Conv2DHDL(HDLGenerator):
    """HDL generator for 2D Convolutional layer."""
    
    def __init__(self, layer: Conv2D, input_width: int = 32, output_width: int = 32,
                 use_pipeline: bool = True):
        """
        Initialize Conv2DHDL generator.
        
        Args:
            layer: Conv2D layer instance
            input_width: Bit width of input values
            output_width: Bit width of output values
            use_pipeline: Whether to use pipelined architecture
        """
        super().__init__(f"conv2d_{layer.out_channels}x{layer.in_channels}x{layer.kernel_size[0]}x{layer.kernel_size[1]}", 
                        input_width, output_width)
        self.layer = layer
        self.use_pipeline = use_pipeline
        self.kernels = layer.kernels.data
        self.bias = layer.bias.data if layer.bias is not None else None
        self.stride = layer.stride
        self.padding = layer.padding
        
    def generate_logic(self) -> None:
        """Generate HDL logic for Conv2D layer."""
        out_channels, in_channels, kernel_h, kernel_w = self.kernels.shape

        self.add_parameter("IN_CHANNELS", in_channels)
        self.add_parameter("OUT_CHANNELS", out_channels)
        self.add_parameter("KERNEL_HEIGHT", kernel_h)
        self.add_parameter("KERNEL_WIDTH", kernel_w)
        self.add_parameter("STRIDE_H", self.stride[0])
        self.add_parameter("STRIDE_W", self.stride[1])
        self.add_parameter("PADDING_H", self.padding[0])
        self.add_parameter("PADDING_W", self.padding[1])
        self.add_parameter("DATA_WIDTH", self.input_width)

        self.add_port("clk", "input")
        self.add_port("rst", "input")
        self.add_port("input_valid", "input")
        self.add_port("input_height", "input", width=16)
        self.add_port("input_width", "input", width=16)
        self.add_port("input_data", "input", width=f"DATA_WIDTH*IN_CHANNELS")
        self.add_port("output_valid", "output")
        self.add_port("output_data", "output", width=f"DATA_WIDTH*OUT_CHANNELS")

        if self.use_pipeline:
            self.add_internal_signal("pipeline_valid", width=4)
            self.add_internal_signal("window_buffer", width=f"DATA_WIDTH*IN_CHANNELS*KERNEL_HEIGHT*KERNEL_WIDTH")
            self.add_internal_signal("accumulator", width=f"DATA_WIDTH*OUT_CHANNELS")

        for oc in range(out_channels):
            for ic in range(in_channels):
                for kh in range(kernel_h):
                    kernel_row = ", ".join([f"{w:.6f}" for w in self.kernels[oc, ic, kh, :]])
                    self.add_parameter(f"KERNEL_{oc}_{ic}_{kh}", f"'{{{kernel_row}}}")
            
            if self.bias is not None:
                self.add_parameter(f"BIAS_{oc}", f"{self.bias[oc]:.6f}")

        if self.use_pipeline:
            input_reg = [
                "if (rst) begin",
                "    pipeline_valid[0] <= 0;",
                "end else begin",
                "    pipeline_valid[0] <= input_valid;",
                "    // Window buffering logic here",
                "end"
            ]
            self.add_always_block(["posedge clk"], input_reg)

            mult_statements = []
            for oc in range(out_channels):
                mult_str = []
                for ic in range(in_channels):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            idx = f"{ic}*KERNEL_HEIGHT*KERNEL_WIDTH + {kh}*KERNEL_WIDTH + {kw}"
                            mult_str.append(
                                f"window_buffer[{idx}*DATA_WIDTH +: DATA_WIDTH] * "
                                f"KERNEL_{oc}_{ic}_{kh}[{kw}]"
                            )
                mult_statements.append(
                    f"accumulator[{oc}*DATA_WIDTH +: DATA_WIDTH] <= {' + '.join(mult_str)};"
                )

            compute = [
                "if (rst) begin",
                "    pipeline_valid[1] <= 0;",
                "end else begin",
                "    pipeline_valid[1] <= pipeline_valid[0];",
                *mult_statements,
                "end"
            ]
            self.add_always_block(["posedge clk"], compute)

            output_statements = []
            for oc in range(out_channels):
                if self.bias is not None:
                    output_statements.append(
                        f"output_data[{oc}*DATA_WIDTH +: DATA_WIDTH] <= "
                        f"accumulator[{oc}*DATA_WIDTH +: DATA_WIDTH] + BIAS_{oc};"
                    )
                else:
                    output_statements.append(
                        f"output_data[{oc}*DATA_WIDTH +: DATA_WIDTH] <= "
                        f"accumulator[{oc}*DATA_WIDTH +: DATA_WIDTH];"
                    )

            output = [
                "if (rst) begin",
                "    pipeline_valid[2] <= 0;",
                "    output_valid <= 0;",
                "end else begin",
                "    pipeline_valid[2] <= pipeline_valid[1];",
                "    output_valid <= pipeline_valid[2];",
                *output_statements,
                "end"
            ]
            self.add_always_block(["posedge clk"], output)
        else:
            output_statements = []
            for oc in range(out_channels):
                mult_str = []
                for ic in range(in_channels):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            idx = f"{ic}*KERNEL_HEIGHT*KERNEL_WIDTH + {kh}*KERNEL_WIDTH + {kw}"
                            mult_str.append(
                                f"input_data[{idx}*DATA_WIDTH +: DATA_WIDTH] * "
                                f"KERNEL_{oc}_{ic}_{kh}[{kw}]"
                            )
                if self.bias is not None:
                    output_statements.append(
                        f"assign output_data[{oc}*DATA_WIDTH +: DATA_WIDTH] = "
                        f"{' + '.join(mult_str)} + BIAS_{oc};"
                    )
                else:
                    output_statements.append(
                        f"assign output_data[{oc}*DATA_WIDTH +: DATA_WIDTH] = "
                        f"{' + '.join(mult_str)};"
                    )

            for stmt in output_statements:
                self.add_assignment(stmt.split("=")[0].strip(), stmt.split("=")[1].strip()[:-1])

            self.add_assignment("output_valid", "input_valid")