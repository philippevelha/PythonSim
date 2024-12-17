# Matrix Multiplication Performance Simulation Toolkit

## Overview

This repository contains advanced Python scripts for simulating and analyzing matrix multiplication performance, with a focus on understanding cache hierarchy, tiling strategies, and computational acceleration techniques. The tools provide a detailed simulation of matrix multiplication that goes beyond traditional benchmarking by incorporating realistic memory access patterns and hardware-specific optimizations.

## Background

Matrix multiplication is a fundamental computational operation in many scientific computing, machine learning, and signal processing applications. The performance of this operation depends critically on:
- Memory hierarchy (L1, L2 caches, main memory)
- Data access patterns
- Computational acceleration techniques
- Tiling and parallel execution strategies

## Scripts

### 1. Latency Matrix Multiplication with Accelerator (`latencyMatrixMultiplicationAcc.py`)

#### Key Features
- Simulates matrix multiplication with a detailed memory hierarchy model
- Incorporates an accelerator for dot product operations
- Uses tiling to optimize cache utilization
- Tracks multiple performance metrics:
  - Total computation time
  - Memory access patterns (L1 and L2 cache)
  - Operation count
  - Accelerator offloading

#### Simulation Parameters
- Configurable latency for different memory levels
- Probabilistic cache access model
- Support for various matrix sizes
- Detailed statistical analysis and visualization

#### Outputs
- Generates log-log plot of computation time vs. matrix size
- Creates statistical visualizations (violin and box plots)
- Saves simulation results as JSON files for further analysis

### 2. Parallel Tiled Matrix Multiplication (`latencyMatrixMultiplicationTilesParallel.py`)

#### Key Features
- Implements a parallel tiling approach for matrix multiplication
- Simulates computation across different parallel execution factors
- Models memory hierarchy impact on performance
- Tracks computational metrics

#### Simulation Parameters
- Configurable tile sizes
- Multiple parallel execution factors
- Detailed memory access simulation
- Cycle-accurate performance modeling

#### Outputs
- Performance results for different matrix sizes and parallel factors
- Detailed metrics on memory access and computational intensity

## Simulation Parameters

Common parameters across both scripts include:
- Memory latency (L1, L2, main memory)
- Cache sizes
- Floating-point data size
- Operational timing characteristics

## Use Cases

These simulation tools are valuable for:
- Computer architecture research
- Performance optimization studies
- Understanding cache and memory hierarchy impacts
- Developing efficient matrix multiplication strategies

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Time module

## How to Use

1. Clone the repository
2. Install required dependencies
3. Run the scripts directly or import functions for custom analysis

```bash
pip install numpy matplotlib
python latencyMatrixMultiplicationAcc.py
python latencyMatrixMultiplicationTilesParallel.py
```

## Future Improvements

- Add more detailed accelerator models
- Implement more advanced tiling strategies
- Create visualization tools for performance analysis
- Support for different matrix types and sparsity

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss potential improvements or extensions to the simulation toolkit.

## License

MIT (2024)

## Authors

Philippe Velha, University of Trento

## Acknowledgments

Next4EXA:  https://eurohpc-ju.europa.eu/net4exa-advancing-european-interconnect-hpc-and-ai-2024-12-13_en

