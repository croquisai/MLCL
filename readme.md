# MLCL

MLCL is a OpenCL-focused machine learning library with the following design goals:
- Small install size
- Easy to use
- Fast
- Flexible
- OpenCL-based for GPU acceleration
- JIT for CPUs (if OpenCL isn't supported!)

Also features a highly experimental HDL generator that can convert a MLCL model into Verilog. (FPGA-friendly architecture!)

## Installation

For now, you will need to pull the repository and run `pip install .` from the root directory.

```bash
git clone https://github.com/rndmcoolawsmgrbg/MLCL.git
cd MLCL
pip install .
```

