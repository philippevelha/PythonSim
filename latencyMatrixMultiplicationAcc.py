import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

# Define cache, memory, and accelerator properties

Lmem = 100   # Latency for main memory in cycles
L2mem = 50   # Latency for L2 cache in cycles
L1mem = 5    # Latency for L1 cache in cycles
Lregister = L1mem # Latency for buffer register near/inside DAC
Lconversion = 20 # Latency for data conversion
L2L1rateTransfer = 64 #bytes per cycle
M = 16          # Minimum size of vector for accelerator
Lacc = 2 + Lconversion  + Lregister   # Latency for accelerator dot product in cyclesLconversion of memory access
L2_size = 2**16  # L2 cache size in bytes
L1_size = 2**14  # L1 cache size in bytes
LoadingTimeL2L1 = L2mem*L1_size/L2L1rateTransfer # time to transfer from L2 to L1 x number of transfer
float_size = 1  # Size of a float in bytes (assuming 32-bit float)
pL1 = 0.001 # probability that the data is not in L1 and needs to be seek in L2

parameters = {'L1mem':L1mem,'L2mem':L2mem,'Lmem':Lmem,'Lregister':Lregister,'Lconversion':Lconversion,'L2L1rateTransfer':L2L1rateTransfer,'Lacc':Lacc,'L2_size':L2_size,'L1_size':L1_size,'float_size':float_size,'pL1':pL1}

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
            # indent=4 makes the JSON file human-readable with proper indentation
            json.dump(dictionary, json_file, indent=4)
        print(f"JSON data successfully saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the JSON file: {e}")

def compute_time_with_tiling_and_accelerator(N):
    # Determine tile size based on L1 cache size
    tile_elements = N**2 // L1_size  # Max elements in a tile for L1 cache
    tile_size = int(np.sqrt(L1_size))  # Tile dimension (tile_size x tile_size)
    # print(f'Tile number: {tile_elements} of size {tile_size} elements')
    # Initialize matrices A and B
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros((N, N))

    # Track total time in cycles
    total_time = 0
    offloaded = 0
    op = 0
    accessmemL1 = 0
    accessmemL2 = 0
    miss = 0

    # Matrix multiplication simulation with cache hierarchy, tiling, and accelerator usage
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, N, tile_size):
                # For each tile, simulate loading into L1 cache
                total_time += LoadingTimeL2L1
                accessmemL2 += L1_size/L2L1rateTransfer
                for ii in range(i, min(i + tile_size, N)):
                    for jj in range(j, min(j + tile_size, N)):
                        op += 1
                        # Use accelerator if the vector size allows (i.e., if M fits within the remaining dimensions)
                        if N - k >= M:
                            # Offload to accelerator
                            C[ii, jj] += np.dot(A[ii, k:k + M], B[k:k + M, jj])
                            if np.random.random(size=1)>pL1:
                                total_time += Lacc
                                accessmemL1 += 1
                                offloaded += 1
                            else:
                                latency = L2mem
                                accessmemL2 += 1
                                miss += 1
                        else:
                            # Fallback to manual multiplication with cache hierarchy if M does not fit
                            for kk in range(k, min(k + tile_size, N)):
                                # Check L1 cache hit
                                if ((ii - i) * tile_size + (kk - k)) < tile_elements or \
                                   ((kk - k) * tile_size + (jj - j)) < tile_elements:
                                    if np.random.random(size=1)>pL1:
                                        latency = L1mem
                                        accessmemL1 +=1
                                        
                                    else: # if the L1 data is not valid or not present then we need to seek the copy in L2
                                        latency = L2mem
                                        accessmemL2 += 1
                                        
                                # Check L2 cache hit if not in L1
                                elif ((ii - i) * tile_size + (kk - k)) < (L2_size // float_size) or \
                                     ((kk - k) * tile_size + (jj - j)) < (L2_size // float_size):
                                    latency = L2mem
                                    accessmemL2 += 1
                                    
                                # Access main memory if not in L1 or L2
                                else:
                                   
                                    latency = Lmem
                                
                                # Perform multiplication and accumulate latency
                                C[ii, jj] += A[ii, kk] * B[kk, jj]
                                total_time += latency
    
    return total_time, accessmemL1, accessmemL2, op, offloaded, miss

# Run simulation for different matrix sizes
matrix_sizes = [4,16,32,64, 128, 256, 512]  # Test for different values of N
results = {}
L1access = {}
L2access = {}
offL = {}
Miss = {}
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
    print(f"Matrix Size: {N}x{N}, computing intensity:{op/total_time}, missed access: {missed}") 

plt.figure(1)
plt.loglog(list(results.keys()),list(results.values()),'--o')
plt.show(block=False)
plt.xlabel('Matrix Size [NxN]')
plt.ylabel('Number of Cycles [#]')
print("----------- STATISITCS -----------")

NBstat = 10
data = []
mm = []
for ii in range(NBstat):
    total_time, accessmemL1, accessmemL2, op, offloaded, missed = compute_time_with_tiling_and_accelerator(128)
    data.append(total_time)
    mm.append(missed)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 4))

axs[0][0].violinplot(data,
                  showmeans=True,
                  showmedians=True)
axs[0][0].set_title('Violin plot')

# plot box plot
axs[0][1].boxplot(data)
axs[0][1].set_title('Box plot')

axs[1][0].violinplot(mm,
                  showmeans=True,
                  showmedians=True)
axs[1][0].set_title('Violin plot')

# plot box plot
axs[1][1].boxplot(mm)
axs[1][1].set_title('Box plot')


plt.show()
ResultTotal = {**parameters , **results }
ResultL1access = {**parameters , **L1access }
ResultL2access = {**parameters , **L2access }
ResultOFLOADED = {**parameters , **offL }
REsultMiss = {**parameters , **Miss}
rndnb = np.random.random()

FILETOTAL = f'total_{int(rndnb*10000)}.json' # change accordingly
FILEL1= f'L1_{int(rndnb*10000)}.json' # change accordingly
FILEL2= f'L2_{int(rndnb*10000)}.json' # change accordingly
FILE_OFFLOAD = f'Offload_{int(rndnb*10000)}.json' # change accordingly
FILE_MISS = f'Miss_{int(rndnb*10000)}.json'
PATHTOTAL= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILETOTAL) #Adjust path accordingly
PATHL1= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILEL1) #Adjust path accordingly
PATHL2= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILEL2) #Adjust path accordingly
PATHOFFLOAD= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILE_OFFLOAD) #Adjust path accordingly
PATHMISS= os.path.join('C:\\Users\\heltz\\Documents\\Research\\', FILE_MISS) #Adjust path accordingly

dict_to_json_file(ResultTotal, filename= PATHTOTAL)
dict_to_json_file(ResultL1access, filename= PATHL1)
dict_to_json_file(ResultL2access, filename= PATHL2)
dict_to_json_file(ResultOFLOADED, filename= PATHOFFLOAD)
dict_to_json_file(REsultMiss, filename= PATHMISS)