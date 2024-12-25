import pyopencl as cl
import numpy as np
import os
from .cpu_ops import cpu_ops

class OpenCLManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenCLManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialize()
            OpenCLManager._initialized = True

    def _initialize(self):
        try:
            platforms = cl.get_platforms()
            for platform in platforms:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    self.ctx = cl.Context(devices=[devices[0]])
                    self.use_opencl = True
                    break
            else:
                for platform in platforms:
                    devices = platform.get_devices(device_type=cl.device_type.CPU)
                    if devices:
                        self.ctx = cl.Context(devices=[devices[0]])
                        self.use_opencl = True
                        break
                else:
                    self.use_opencl = False
                    print("Warning: No OpenCL devices found. Falling back to CPU with JIT acceleration.")
                    return
        except:
            self.use_opencl = False
            print("Warning: OpenCL initialization failed. Falling back to CPU with JIT acceleration.")
            return

        if self.use_opencl:
            self.queue = cl.CommandQueue(self.ctx)
            self.program_cache = {}
            self._load_kernels()

    def _load_kernels(self):
        if not self.use_opencl:
            return
            
        kernel_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kernels')
        for filename in os.listdir(kernel_dir):
            if filename.endswith('.cl'):
                with open(os.path.join(kernel_dir, filename), 'r') as f:
                    source = f.read()
                    program_name = os.path.splitext(filename)[0]
                    self.program_cache[program_name] = cl.Program(self.ctx, source).build()

    def get_program(self, source_code=None, program_name="default"):
        if not self.use_opencl:
            return None
            
        if source_code is None:
            if program_name not in self.program_cache:
                raise ValueError(f"No program found with name {program_name}")
            return self.program_cache[program_name]
        
        cache_key = hash(source_code)
        if cache_key not in self.program_cache:
            self.program_cache[cache_key] = cl.Program(self.ctx, source_code).build()
        return self.program_cache[cache_key]

    def get_ops(self):
        """Return the appropriate operations manager based on availability."""
        if self.use_opencl:
            from .opencl_ops import opencl_ops
            return opencl_ops
        return cpu_ops

opencl_manager = OpenCLManager() 