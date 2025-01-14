import math
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os




@dataclass
class OperationCounts:
    flops: int  # Floating point operations
    reads: int  # Memory reads
    writes: int # Memory writes
    
def count_fft(N: int) -> OperationCounts:
    """Calculate operations for FFT (Cooley-Tukey algorithm)"""
    stages = int(math.log2(N))
    butterflies_per_stage = N // 2
    # Each butterfly: 6 flops (4 mults + 2 adds), 4 reads, 2 writes
    flops = stages * butterflies_per_stage * 6
    reads = stages * butterflies_per_stage * 4
    writes = stages * butterflies_per_stage * 2
    return OperationCounts(flops, reads, writes)

def count_gemm(N: int) -> OperationCounts:
    """Calculate operations for matrix multiplication (N x N matrices)"""
    # For each element: N multiplications and N-1 additions
    flops = N * N * (2*N - 1)  # N^2 elements, each needs N mults and N-1 adds
    reads = 2 * N * N * N  # Each element of both matrices read N times
    writes = N * N  # Each element of result matrix written once
    return OperationCounts(flops, reads, writes)

def count_convolution(N: int, K: int = 3) -> OperationCounts:
    """Calculate operations for 2D convolution with kernel size K"""
    output_size = N - K + 1
    # For each output element: K*K multiplications and K*K-1 additions
    flops = output_size * output_size * (2 * K * K - 1)
    reads = output_size * output_size * K * K + K * K  # Input window + kernel
    writes = output_size * output_size
    return OperationCounts(flops, reads, writes)

def count_svd(N: int) -> OperationCounts:
    """Approximate operations for SVD (assuming N x N matrix)"""
    # Rough approximation based on QR iteration
    iterations = 10  # Typical number of iterations
    flops = iterations * (4 * N * N * N)  # Dominant term
    reads = iterations * (3 * N * N)
    writes = iterations * (3 * N * N)
    return OperationCounts(flops, reads, writes)

def count_nbody(N: int) -> OperationCounts:
    """Calculate operations for N-body simulation (single timestep)"""
    # For each pair: 3D distance calc (9 flops) + force calc (6 flops)
    pairs = (N * (N-1)) // 2
    flops = pairs * 15
    reads = pairs * 6  # Position vectors for each pair
    writes = N * 3  # Update force vectors
    return OperationCounts(flops, reads, writes)

def count_monte_carlo(N: int) -> OperationCounts:
    """Calculate operations for Monte Carlo integration (N samples)"""
    # Per sample: random number gen (4 flops) + function eval (10 flops) + accumulate (1 flop)
    flops = N * 15
    reads = N * 2  # Read random numbers and previous sum
    writes = N  # Update running sum
    return OperationCounts(flops, reads, writes)

def count_eigendecomp(N: int) -> OperationCounts:
    """Approximate operations for eigenvalue decomposition"""
    # Based on QR algorithm, similar to SVD but fewer operations
    iterations = 10
    flops = iterations * (3 * N * N * N)
    reads = iterations * (2 * N * N)
    writes = iterations * (2 * N * N)
    return OperationCounts(flops, reads, writes)

def count_viterbi(N: int, T: int = None) -> OperationCounts:
    """Calculate operations for Viterbi algorithm (N states, T sequence length)"""
    if T is None:
        T = N
    # For each t,j: max over N previous states + transition
    flops = T * N * (2 * N)  # N comparisons and N additions per state
    reads = T * N * (2 * N)  # Read previous probs and transitions
    writes = T * N  # Store best path probabilities
    return OperationCounts(flops, reads, writes)

def count_lu_decomp(N: int) -> OperationCounts:
    """Calculate operations for LU decomposition"""
    # Based on Doolittle algorithm
    flops = (2 * N * N * N) // 3  # Approximate dominant term
    reads = N * N * N // 2
    writes = N * N
    return OperationCounts(flops, reads, writes)

def count_dct(N: int) -> OperationCounts:
    """Calculate operations for 2D DCT (N x N block)"""
    # Similar to FFT but with real numbers
    flops = 2 * N * N * math.log2(N)  # Approximate for 2D
    reads = 2 * N * N
    writes = N * N
    return OperationCounts(flops, reads, writes)

def print_operation_counts(N: int):
    """Print operation counts for all algorithms"""
    algorithms = {
        "FFT": count_fft,
        "GeMM": count_gemm,
        "Convolution": count_convolution,
        "SVD": count_svd,
        "N-body": count_nbody,
        "Monte Carlo": count_monte_carlo,
        "Eigendecomp": count_eigendecomp,
        "Viterbi": count_viterbi,
        "LU Decomp": count_lu_decomp,
        "DCT": count_dct
    }
    
    print(f"Operation counts for N = {N}")
    print("-" * 80)
    print(f"{'Algorithm':<15} {'FLOPs':<15} {'Reads':<15} {'Writes':<15} {'Total Memory':<15}")
    print("-" * 80)
    
    for name, func in algorithms.items():
        counts = func(N)
        total_memory = counts.reads + counts.writes
        print(f"{name:<15} {counts.flops:<15} {counts.reads:<15} {counts.writes:<15} {total_memory:<15}")
        rndnb = np.random.random()
        FILENAME = f'Pie_{int(rndnb*10000)}.png'
        PATHTOTAL= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILENAME) #Adjust path accordingly
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        #plt.figure()
        fig, ax = plt.subplots()
        ax.pie([counts.flops,counts.reads,counts.writes,total_memory], explode=explode, labels=['FLOPS','Reads','Writes','Mem Access'], autopct='%1.1f%%',
            shadow=True, startangle=90)
        plt.show(block=False)
        plt.title(f'Algorithm: {name} \n {N} Elements ')
        plt.savefig(PATHTOTAL)
    #plt.show(block=True)

# Example usage
listN = [256,512,1024,2048,4096,2**13,2**14,2**15,2**16]
for N in listN:
    print_operation_counts(N)