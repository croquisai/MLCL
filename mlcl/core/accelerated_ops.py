from .opencl_utils import opencl_manager

ops_manager = opencl_manager.get_ops()

accelerated_ops = ops_manager

"old compatibility file"