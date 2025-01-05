# Contributing Guide

Thank you for your interest in contributing to MLCL! We welcome contributions from the community to help improve the framework and make it more accessible to users.

Before contributing, please review the following guidelines to ensure your changes align with the project's goals and coding standards. (If it doesn't, we can try fix it up for you! 😉)

## Code Style Guidelines

1. **Readability First**
   - Use clear, descriptive variable and function names
   - Write code that is easy to read and understand
   - Follow the PEP 8 style guide

2. **Implementation Requirements** 
   - Keep code lightweight and efficient
   - Avoid unnecessary dependencies (e.g., TQDM for progress bars, where it could be easily replaced with a function)
   - Benchmark performance-critical code to ensure it is optimized
   - Use NumPy arrays for multidimensional data
   - Use Numba for JIT-compiled functions

3. **Documentation**
   - Document all public functions and classes
   - Include usage examples where appropriate
   - Keep documentation up-to-date with code changes
   - Add references for complex algorithms

4. **Pull Request Process**
   - Create a branch for your changes
   - Write clear commit messages
   - Update documentation as needed
   - Add tests for new features
   - Request review from maintainers

## Getting Started

1. Fork the repository
2. Create your feature branch
3. Commit your changes 
4. Push to your fork
5. Open a Pull Request