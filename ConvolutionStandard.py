import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

class LatencySimulator:
    def __init__(self,
                 l1_cache_size=1*1024,  # 4 KB L1 cache
                 l2_cache_size=16*1024,  # 16 KB L2 cache
                 memory_read_latency=500,  # Main memory read latency (cycles)
                 l1_read_latency=5,      # L1 cache read latency (cycles)
                 l2_read_latency=50,      # L2 cache read latency (cycles)
                 compute_latency=1, # Basic computation latency (cycles)
                 L2L1rateTransfer = 64 ):      #bytes per cycle
        """
        Initialize latency simulation parameters
       
        Args:
            l1_cache_size: Size of L1 cache in bytes
            l2_cache_size: Size of L2 cache in bytes
            memory_read_latency: Latency for reading from main memory
            l1_read_latency: Latency for reading from L1 cache
            l2_read_latency: Latency for reading from L2 cache
            compute_latency: Latency for basic computational operations
            L2L1rateTransfer: Indicates the number of bytes transferred per cycle from L2 to L1
        """
        self.l1_cache_size = l1_cache_size
        self.l2_cache_size = l2_cache_size
        self.memory_read_latency = memory_read_latency
        self.l1_read_latency = l1_read_latency
        self.l2_read_latency = l2_read_latency
        self.compute_latency = compute_latency
        self.L2L1rateTransfer = L2L1rateTransfer
       
        # Total latency tracking
        self.total_latency = 0
   
    def get_read_latency(self, size_bytes):
        """
        Determine read latency based on data size and cache levels
       
        Args:
            size_bytes: Size of data being read in bytes
       
        Returns:
            Latency for reading the data
        """
        if size_bytes <= self.l1_cache_size:
            return self.l1_read_latency
        elif size_bytes <= self.l2_cache_size:
            return self.l2_read_latency + int(size_bytes/self.L2L1rateTransfer)*self.l1_read_latency
        else:
            return self.memory_read_latency
   
    def convolution_with_latency(self, matrix, kernel, padding=0):
        """
        Perform convolution with latency simulation
       
        Args:
            matrix: Input matrix (numpy array)
            kernel: Convolution kernel (numpy array)
            padding: Padding size
       
        Returns:
            Convolved matrix with latency information
        """
        # Reset total latency
        self.total_latency = 0
       
        # Matrix and kernel dimensions
        N, M = matrix.shape
        K, L = kernel.shape
       
        # Output matrix dimensions
        output_height = N - K + 1 + 2 * padding
        output_width = M - L + 1 + 2 * padding
        output = np.zeros((output_height, output_width))
       
        # Determine tile sizes based on cache
        element_size = matrix.itemsize
        max_tile_rows = int(np.sqrt(self.l1_cache_size / (element_size * M)))
        max_tile_rows = max(max_tile_rows, K)  # Ensure kernel fits
       
        # Tiled convolution
        for i in range(0, N, max_tile_rows):
            for j in range(0, M, max_tile_rows):
                # Tile extraction with latency
                tile_height = min(max_tile_rows, N - i)
                tile_width = min(max_tile_rows, M - j)
               
                # Read tile latency
                tile_size = tile_height * tile_width * element_size
                self.total_latency += self.get_read_latency(tile_size)
               
                # Convolution for this tile
                for h in range(output_height):
                    for w in range(output_width):
                        # Extract sub-region for convolution
                        region = matrix[
                            max(0, h-padding):min(N, h+K-padding),
                            max(0, w-padding):min(M, w+L-padding)
                        ]
                       
                        # Convolution computation with latency
                        output[h, w] += np.sum(region * kernel)
                        self.total_latency += self.compute_latency * K * L
       
        return output, self.total_latency

def main():
    # Create latency simulator
    simulator = LatencySimulator()
   
    # Example matrices
    matrix_sizes = [16,32, 64, 128, 256]
    kernel_sizes = [3, 5, 7]
    Latency3 = {}
    Latency5 = {}
    Latency7 = {}
    for N in matrix_sizes:
        
        for K in kernel_sizes:
            # Create random matrix and kernel
            matrix = np.random.rand(N, N)
            kernel = np.random.rand(K, K)
           
            # Perform convolution with latency simulation
            print(f"\nMatrix Size: {N}x{N}, Kernel Size: {K}x{K}")
            result, total_latency = simulator.convolution_with_latency(matrix, kernel)
            if K == 3:
                Latency3[N] = total_latency
            if K == 5:
                Latency5[N] = total_latency
            if K == 7:
                Latency7[N] = total_latency

            print(f"Total Latency Cycles: {total_latency}")
            print(f"Output Matrix Shape: {result.shape}")
    ax = plt.axes() 
    plt.loglog(list(Latency3.keys()),list(Latency3.values()),'--o')
    plt.loglog(list(Latency5.keys()),list(Latency5.values()),'--s')
    plt.loglog(list(Latency7.keys()),list(Latency7.values()),'--x')
    #plt.legend('[3x3]','[5x5]','[7x7]')
    # setting ticks for x-axis 
    ax.set_xticks(matrix_sizes) 
    ax.set_xticklabels(list(Latency3.keys())) 
    ax.minorticks_off()
    plt.xlabel('Matrix Size [NxN]')
    plt.ylabel('Number of Cycles [#]')
    plt.show(block=True)  
    
    time.sleep(1)
if __name__ == "__main__":
    main()