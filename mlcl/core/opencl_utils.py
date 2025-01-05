import pyopencl as cl
import os
import numpy as np
from typing import Optional, Dict, Any
import logging
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
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("OpenCLManager")

            self.ctx = None
            self.queue = None
            self.device = None
            self.use_opencl = False
            self.program_cache = {}
            self.device_info = {}
            
            self._initialize()
            OpenCLManager._initialized = True
    
    def _select_best_device(self, devices: list) -> Optional[cl.Device]:
        """Select the best device based on compute capabilities."""
        if not devices:
            return None

        device_scores = []
        for device in devices:
            score = 0
            try:
                score += device.max_compute_units * 100
                score += device.global_mem_size / (1024 * 1024)
                score += getattr(device, 'max_clock_frequency', 0)
                score += device.max_work_group_size / 1024
                
                device_scores.append((score, device))
            except cl.LogicError:
                continue
        
        if not device_scores:
            return None
        return max(device_scores, key=lambda x: x[0])[1]
    
    def _initialize(self):
        """Initialize OpenCL context and command queue."""
        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            for platform in platforms:
                self.logger.info(f"Checking platform: {platform.name}")

                try:
                    devices = platform.get_devices(device_type=cl.device_type.GPU)
                    device = self._select_best_device(devices)
                    if device:
                        self.device = device
                        self.logger.info(f"Selected GPU device: {device.name}")
                        self.logger.info(f"Max work group size: {device.max_work_group_size}")
                        self.logger.info(f"Max compute units: {device.max_compute_units}")
                        self.logger.info(f"Local memory size: {device.local_mem_size}")
                        self.logger.info(f"Global memory size: {device.global_mem_size}")
                        break
                except cl.LogicError:
                    continue

            if not self.device:
                for platform in platforms:
                    try:
                        devices = platform.get_devices(device_type=cl.device_type.CPU)
                        device = self._select_best_device(devices)
                        if device:
                            self.device = device
                            self.logger.info(f"Selected CPU device: {device.name}")
                            break
                    except cl.LogicError:
                        continue
            
            if not self.device:
                raise RuntimeError("No suitable OpenCL devices found")

            self.ctx = cl.Context(devices=[self.device])
            self.queue = cl.CommandQueue(self.ctx, 
                                       properties=cl.command_queue_properties.PROFILING_ENABLE)
            
            self.use_opencl = True
            self._load_kernels()
            
        except Exception as e:
            self.logger.error(f"OpenCL initialization failed: {str(e)}")
            self.use_opencl = False
            self.logger.warning("Falling back to CPU with JIT acceleration")
    
    def _load_kernels(self):
        """Load and compile OpenCL kernels with caching."""
        if not self.use_opencl:
            return
            
        kernel_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'kernels')

        for filename in os.listdir(kernel_dir): # this might be a bad idea, security-wise, memory-wise and loadtime-wise. might try come up with a better solution
            if filename.endswith('.cl'):
                try:
                    with open(os.path.join(kernel_dir, filename), 'r') as f:
                        source = f.read()

                    program_name = os.path.splitext(filename)[0]
                    build_options = [
                        "-cl-mad-enable",
                        "-cl-fast-relaxed-math",
                        "-cl-no-signed-zeros",
                        "-cl-denorms-are-zero",
                    ]
                    
                    program = cl.Program(self.ctx, source)
                    program.build(options=' '.join(build_options))
                    self.program_cache[program_name] = program
                    
                    self.logger.info(f"Successfully loaded kernel(s): {program_name}")
                    
                except cl.LogicError as e:
                    self.logger.error(f"Failed to build kernel {filename}: {str(e)}")
                    if hasattr(program, 'get_build_info'):
                        self.logger.error(program.get_build_info(self.device, cl.program_build_info.LOG))


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