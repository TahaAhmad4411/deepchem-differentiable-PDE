# Differentiable FEM Solver (PyTorch)

This project implements a differentiable finite element method (FEM) solver using PyTorch.

## Features
- 2D Poisson solver
- Mesh data structure
- Differentiable pipeline (autograd-compatible)
- Benchmarking (h-convergence, inverse problem)

## Results
- Convergence rate ~ O(h²)
- Successful inverse problem optimization (>99% loss reduction)

## Next Steps
- Add visualization
- Compare with standard FEM solvers
