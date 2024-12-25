from mlcl.nn.layers import Linear
from mlcl.hdl import ModelToHDLPipeline
from mlcl import ModelIO

model_io = ModelIO()

model = [
    Linear(2, 8),
    Linear(8, 1)
]

for layer in model:
    model_io.apply("xor_model", [layer.weights, layer.bias])

pipeline = ModelToHDLPipeline(output_dir="hdl_out")
hdl_files = pipeline.convert_model(model, model_name="showcase")