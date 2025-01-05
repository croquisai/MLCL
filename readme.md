# MLCL

MLCL is a lightweight deep learning framework with cross-platform GPU acceleration using OpenCL. It is designed to be easy to use and to provide a high level of control over the GPU. MLCL is built on top of the [PyOpenCL](https://documen.tician.de/pyopencl/) library.

## Installation
```bash
git clone https://github.com/open-gpgpu/MLCL.git
cd MLCL
pip install -e .
```

## Usage
Documentation is available at [docs](/docs.md).

## Interesting Features
- Supports both CPU and GPU acceleration
- Supports automatic differentiation
- Supports saving and loading models
(those are all normal features of PyTorch, but with a few extra tricks up its sleeve!)
- Scores GPUs for the best performance
- Built-in HDL conversion for FPGAs and ASICs (whoa...)
- Streaming datasets while training for slower cards (like mining cards!)

## Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](/CONTRIBUTING.md) for more information.

## License
MLCL is licensed under the [GNU General Public License v3.0](/LICENSE).
(that practically means, you can use it for commercial purposes, but you have to give me credit and you can't sell it)