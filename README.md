# Implicit Neural Networks - Learning Implementations

My implementations of implicit neural networks while learning the concepts.

## Background

These notebooks follow the paper, and cnapters 1 and 4 from the NeurIPS 2020 tutorial:
- **Tutorial:** https://implicit-layers-tutorial.org/
- **Paper:** Bai et al. "Deep Equilibrium Models" (NeurIPS 2019)

## Structure

**Fixed Point Iteration Layer** (`fixed_point_iteration_layer.ipynb`)
- Basic implicit layer using fixed-point iteration
- Solves $z* = \sigma(Wz^* + x)$ by repeatedly applying the function
- Simple but slow convergence

**Implicit Newton Layer** (`implicit_newton_layer.ipynb`)
- Faster convergence using Newton's method
- Uses Jacobian information for smarter updates
- 7.3x faster than fixed-point iteration

**Implicit Differentiation** (`implicit_differentiation.ipynb`)
- Memory-efficient backpropagation through implicit layers
- Uses implicit function theorem instead of storing Jacobians

**Deep Equilibrium Model** (`deep_equilibrium_model.ipynb`)
- Full DEQ implementation with Anderson acceleration
- Infinite-depth network with finite compute
- Uses torchdeq library for Anderson acceleration

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Open any notebook and run the cells. Each is self-contained.

## Notes
These are personal learning notebooks, not production code. I'm implementing the concepts myself to understand them better.