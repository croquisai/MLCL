from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlcl",
    version="0.1.2",
    author="open-gpgpu",
    author_email="ilovevisualstudiocode@gmail.com",
    description="A lightweight deep learning framework with cross-platform GPU acceleration using OpenCL.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/open-gpgpu/MLCL",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pyopencl",
        "pybind11",
        "pydantic",
        "pydantic-settings",
        "pydantic-core",
        "numba",
        "siphash24",
    ],
)
