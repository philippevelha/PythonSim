import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

# Define cache, memory, and accelerator properties
Lmem = 400   # Latency for main memory in cycles
L2mem = 50   # Latency for L2 cache in cycles
L1mem = 20   # Latency for L1 cache in cycles
Lregister = L1mem  # Latency for buffer register near/inside DAC
Lconversion = 2  # Latency for data conversion
L2L1rateTransfer = 64  # Bytes per cycle
M = 16    # Minimum size of vector for accelerator
Lacc = 2 + Lconversion + Lregister   # Latency for accelerator dot product in cycles
L2_size = 40 * 1024**2  # L2 cache size in bytes
L1_size = 256 * 1024  # L1 cache size in bytes
float_size = 4  # Size of a float in bytes (corrected to 4 bytes for 32-bit float)
pL1 = 0.001  # Probability that the data is not in L1 and needs to be seek in L2

# Calculate time to transfer entire L1 cache from L2
LoadingTimeL2L1 = L2mem + (L1_size / L2L1rateTransfer)  

parameters = {
    'L1mem': L1mem,
    'L2mem': L2mem,
    'Lmem': Lmem,
    'Lregister': Lregister,
    'Lconversion': Lconversion,
    'L2L1rateTransfer': L2L1rateTransfer,
    'Lacc': Lacc,
    'L2_size': L2_size,
    'L1_size': L1_size,
    'float_size': float_size,
    'pL1': pL1,
    'M': M
}

def dict_to_json_file(dictionary, filename):
    """
    Convert a dictionary to JSON and save it in a file.
    
    Parameters:
    dictionary (dict): The dictionary to be converted to JSON
    filename (str): The name of the file to save the JSON data
    
    Returns:
    None
    """
    try:
        # Open the file in write mode
        with open(filename, 'w') as json_file:
            # Use json.dump() to write the dictionary to the file in JSON format
            json.dump(dictionary, json_file, indent=4)
        print(f"JSON data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")

def compute_time_with_tiling_and_accelerator(N):
    """
    Compute the time required for matrix multiplication with tiling and accelerator.
    
    Parameters:
    N (int): The dimension of the square matrices (N×N)
    
    Returns:
    tuple: (total_time, accessmemL1, accessmemL2, op, offloaded, miss)
    """
    # Calculate max elements per tile based on L1 cache size
    # Need to fit 3 matrices (A, B, C) in L1 cache
    max_elements_in_L1 = L1_size // float_size // 3
    
    # Determine tile size (maximum square tile that fits in L1)
    tile_size = int(np.sqrt(max_elements_in_L1))
    
    # Ensure tile size doesn't exceed matrix size
    tile_size = min(N, tile_size)
    
    print(f'Tile size: {tile_size}x{tile_size}')
    
    # Initialize matrices A and B
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))
    
    # Track metrics
    total_time = 0
    offloaded = 0
    op = 0  # Total operations
    accessmemL1 = 0
    accessmemL2 = 0
    miss = 0

    # Matrix multiplication simulation with tiling
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                # For each tile, simulate loading into L1 cache from L2
                # This loads the current tile of A, B into L1
                total_time += LoadingTimeL2L1
                accessmemL2 += 2 * (tile_size * tile_size)  # Load both A and B tiles
                
                # Process the current tiles
                for ii in range(i, min(i + tile_size, N)):
                    for jj in range(j, min(j + tile_size, N)):
                        # Check if we can use accelerator for this row/column combination
                        remaining_k = min(k + tile_size, N) - k
                        
                        if remaining_k >= M:
                            # We can use accelerator for chunks of size M
                            for kk in range(k, min(k + tile_size, N), M):
                                remaining = min(kk + M, min(k + tile_size, N)) - kk
                                
                                if remaining >= M:
                                    # Use accelerator for full M-sized vector
                                    op += M
                                    
                                    # L1 cache hit check with probability
                                    if np.random.random() > pL1:
                                        total_time += Lacc
                                        accessmemL1 += 2 * M  # Access M elements from A and B
                                        offloaded += 1
                                    else:
                                        # L1 cache miss, fetch from L2
                                        total_time += L2mem + Lacc
                                        accessmemL2 += 2 * M  # Fetch M elements from L2
                                        accessmemL1 += 2 * M  # Then access from L1
                                        miss += 1
                                        offloaded += 1
                                    
                                    # Add the dot product result to C[ii, jj]
                                    C[ii, jj] += np.dot(A[ii, kk:kk+M], B[kk:kk+M, jj])
                                else:
                                    # Not enough elements for M, do element-wise multiplication
                                    for k_elem in range(kk, min(kk + remaining, N)):
                                        op += 1
                                        
                                        # Check L1 cache hit
                                        if np.random.random() > pL1:
                                            total_time += L1mem
                                            accessmemL1 += 2  # Access A[ii, k_elem] and B[k_elem, jj]
                                        else:
                                            # L1 cache miss, fetch from L2
                                            total_time += L2mem
                                            accessmemL2 += 2
                                            miss += 1
                                        
                                        # Perform multiplication and accumulate
                                        C[ii, jj] += A[ii, k_elem] * B[k_elem, jj]
                        else:
                            # Can't use accelerator, do element-wise multiplication
                            for kk in range(k, min(k + tile_size, N)):
                                op += 1
                                
                                # Check L1 cache hit
                                if np.random.random() > pL1:
                                    total_time += L1mem
                                    accessmemL1 += 2  # Access A[ii, kk] and B[kk, jj]
                                else:
                                    # L1 cache miss, fetch from L2
                                    total_time += L2mem
                                    accessmemL2 += 2
                                    miss += 1
                                
                                # Perform multiplication and accumulate
                                C[ii, jj] += A[ii, kk] * B[kk, jj]
    
    return total_time, accessmemL1, accessmemL2, op, offloaded, miss

# Run simulation for different matrix sizes
matrix_sizes = [4, 16, 32, 64, 128, 256, 512]  # Test for different values of N
results = {}
L1access = {}
L2access = {}
offL = {}
Miss = {}
Stats = False

for N in matrix_sizes:
    start_time = time.time()
    total_time, accessmemL1, accessmemL2, op, offloaded, missed = compute_time_with_tiling_and_accelerator(N)
    end_time = time.time()
    
    results[N] = total_time
    L1access[N] = accessmemL1
    L2access[N] = accessmemL2
    offL[N] = offloaded
    Miss[N] = missed
    
    print(f"Matrix Size: {N}x{N}, Computation Time with Tiling and Accelerator (cycles): {total_time}, Elapsed Time: {end_time - start_time:.4f} seconds")
    print(f"Matrix Size: {N}x{N}, memory access L1: {accessmemL1}, memory access L2: {accessmemL2}, number of op: {op}, offloaded: {offloaded}")
    print(f"Matrix Size: {N}x{N}, computing intensity: {op/total_time:.4f}, missed access: {missed}")

plt.figure(1)
plt.loglog(list(results.keys()), list(results.values()), '--o')
plt.title('Matrix Multiplication Performance')
plt.xlabel('Matrix Size [NxN]')
plt.ylabel('Number of Cycles [#]')
plt.grid(True, which="both", ls="--")
plt.savefig('matrix_multiplication_performance.png')
plt.show(block=True)

if Stats:
    print("----------- STATISTICS -----------")
    
    NBstat = 100
    data = []
    mm = []
    for ii in range(NBstat):
        total_time, accessmemL1, accessmemL2, op, offloaded, missed = compute_time_with_tiling_and_accelerator(128)
        data.append(total_time)
        mm.append(missed)
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    
    axs[0][0].violinplot(data, showmeans=True, showmedians=True)
    axs[0][0].set_title('Compute Time [cycles]')
    
    # plot box plot
    axs[0][1].boxplot(data)
    axs[0][1].set_title('Box plot')
    
    axs[1][0].violinplot(mm, showmeans=True, showmedians=True)
    axs[1][0].set_title('Miss number')
    
    # plot box plot
    axs[1][1].boxplot(mm)
    axs[1][1].set_title('Box plot')
    
    plt.tight_layout()
    plt.savefig('statistics.png')
    plt.show()

# Store results in JSON files
ResultTotal = {**parameters, **results}
ResultL1access = {**parameters, **L1access}
ResultL2access = {**parameters, **L2access}
ResultOFLOADED = {**parameters, **offL}
ResultMiss = {**parameters, **Miss}

# Generate random number for filenames
rndnb = np.random.random()

# Create filenames
FILETOTAL = f'total_{int(rndnb*10000)}.json'
FILEL1 = f'L1_{int(rndnb*10000)}.json'
FILEL2 = f'L2_{int(rndnb*10000)}.json'
FILE_OFFLOAD = f'Offload_{int(rndnb*10000)}.json'
FILE_MISS = f'Miss_{int(rndnb*10000)}.json'

# Use current directory for output files for portability
current_dir = os.getcwd()
PATHTOTAL = os.path.join(current_dir, FILETOTAL)
PATHL1 = os.path.join(current_dir, FILEL1)
PATHL2 = os.path.join(current_dir, FILEL2)
PATHOFFLOAD = os.path.join(current_dir, FILE_OFFLOAD)
PATHMISS = os.path.join(current_dir, FILE_MISS)

# Save results to JSON files
dict_to_json_file(ResultTotal, filename=PATHTOTAL)
dict_to_json_file(ResultL1access, filename=PATHL1)
dict_to_json_file(ResultL2access, filename=PATHL2)
dict_to_json_file(ResultOFLOADED, filename=PATHOFFLOAD)
dict_to_json_file(ResultMiss, filename=PATHMISS)