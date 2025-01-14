import numpy as np
import time
from math import ceil
import matplotlib.pyplot as plt

# Define cache and memory properties
Lmem = 400   # Latency for main memory in cycles
L2mem = 50   # Latency for L2 cache in cycles
L1mem = 5    # Latency for L1 cache in cycles
L2_size = 10*1024**2  # L2 cache size in bytes
L1_size = 16*1024  # L1 cache size in bytes
float_size = 4   # Size of a float in bytes (assuming 32-bit float)
MACop = 3 #Time to perform a MAC operation in cycles depending can go up to 5 or 6

def compute_time_with_tiling_parallel(N, K):
    # Determine tile size based on L1 cache size
    tile_elements = L1_size // K  # Max elements in a tile for L1 cache
    tile_size = int(np.sqrt(tile_elements))  # Tile dimension (tile_size x tile_size)

    # Initialize matrices A and B
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))

    # Track total time in cycles
    total_time = 0
    op = 0
    accessmemL1 = 0
    accessmemL2 = 0

    # Matrix multiplication simulation with cache hierarchy and parallel tiling
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            # Initialize a list to store the maximum latency for each parallel round
            max_latency_per_round = []
            
            # Split tiles into chunks of K to simulate parallel execution
            for k_group in range(0, N, tile_size * K):
                # For each tile in this parallel group, accumulate max latency
                max_latency = 0
                
                for k in range(k_group, min(k_group + tile_size * K, N), tile_size):
                    # Simulate computation for a single tile
                    tile_latency = 0
                    for ii in range(i, min(i + tile_size, N)):
                        for jj in range(j, min(j + tile_size, N)):
                            for kk in range(k, min(k + tile_size, N)):
                                op +=1
                                
                                # Check L1 cache hit
                                if ((ii - i) * tile_size + (kk - k)) < tile_elements or \
                                   ((kk - k) * tile_size + (jj - j)) < tile_elements:
                                    latency = L1mem
                                    accessmemL1 +=1
                                # Check L2 cache hit if not in L1
                                elif ((ii - i) * tile_size + (kk - k)) < (L2_size // float_size) or \
                                     ((kk - k) * tile_size + (jj - j)) < (L2_size // float_size):
                                    latency = L2mem
                                    accessmemL2 +=1
                                # Access main memory if not in L1 or L2
                                else:
                                    latency = Lmem
                                
                                # Perform multiplication and accumulate latency for this tile
                                C[ii, jj] += A[ii, kk] * B[kk, jj]
                                tile_latency += latency + MACop
                    
                    # Update the maximum latency for the group of K tiles
                    max_latency = max(max_latency, tile_latency)
                
                # Add the maximum latency for this parallel group to the total time
                max_latency_per_round.append(max_latency)
            
            # Sum all maximum latencies across rounds of K-parallel tiles
            total_time += sum(max_latency_per_round)
    
    return total_time, accessmemL1, accessmemL2, op

# Run simulation for different matrix sizes and parallel factors
matrix_sizes = [16,32,64, 128,256, 512]  # Test for different values of N
parallel_factors = [2, 4, 8]   # Test for different values of K
results = {}

for N in matrix_sizes:
    for K in parallel_factors:
        start_time = time.time()
        total_time, accessmemL1, accessmemL2, op  = compute_time_with_tiling_parallel(N, K)
        end_time = time.time()
        results[(N, K)] = total_time
        print(f"Matrix Size: {N}x{N}, Parallel Factor K={K}, Computation Time with Tiling (cycles): {total_time}, Elapsed Time: {end_time - start_time:.4f} seconds")
        print(f"Matrix Size: {N}x{N}, Parallel Factor K={K}, memory access L1: {accessmemL1}, memory access L2: {accessmemL2}, number of op: {op}")
        print(f"Matrix Size: {N}x{N}, Parallel Factor K={K}, computing intensity:{op/total_time} in op/cycle")             



# Extract data for each second value
data_2 = {key[0]: value for key, value in results.items() if key[1] == 2}
data_4 = {key[0]: value for key, value in results.items() if key[1] == 4}
data_8 = {key[0]: value for key, value in results.items() if key[1] == 8}

# Create lists for plotting
x_values = sorted(list(data_2.keys()))  # [16, 32, 64, 128]
y_values_2 = [data_2[x] for x in x_values]
y_values_4 = [data_4[x] for x in x_values]
y_values_8 = [data_8[x] for x in x_values]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values_2, 'o-', label='K = 2', linewidth=2, markersize=8)
plt.plot(x_values, y_values_4, 's-', label='K = 4', linewidth=2, markersize=8)
plt.plot(x_values, y_values_8, '^-', label='K = 8', linewidth=2, markersize=8)

# Customize the plot
plt.xlabel('Matrix Size [NxN]')
plt.ylabel('Number of cycles')
plt.title('Performance Comparison')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Use logarithmic scale for better visualization
plt.xscale('log', base=2)
plt.yscale('log', base=10)


plt.show(block=True)
