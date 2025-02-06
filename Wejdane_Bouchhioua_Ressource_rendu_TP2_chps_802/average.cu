#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>

// Constants in constant memory
__constant__ float a, b, c;

#define THREADS_PER_BLOCK 128
#define NUM_VALUES_PER_THREAD 200

__global__ void compute_average(float *d_results, int n_threads) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n_threads) return;

    curandState state;
    curand_init(1234, idx, 0, &state); // Initialize random state

    float sum = 0.0f;

    for (int i = 0; i < NUM_VALUES_PER_THREAD; i++) {
        float z = curand_normal(&state); // Generate standard normal random variable
        sum += a * z * z + b * z + c;
    }

    d_results[idx] = sum / NUM_VALUES_PER_THREAD;
}

int main() {
    int num_threads = 1024; // Total number of threads
    int num_blocks = (num_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Host constants
    float h_a = 2.0f, h_b = 1.0f, h_c = 3.0f;

    // Copy constants to constant memory
    cudaMemcpyToSymbol(a, &h_a, sizeof(float));
    cudaMemcpyToSymbol(b, &h_b, sizeof(float));
    cudaMemcpyToSymbol(c, &h_c, sizeof(float));

    // Allocate memory on device
    float *d_results;
    cudaMalloc((void **)&d_results, sizeof(float) * num_threads);

    // Allocate memory on host
    float *h_results = (float *)malloc(sizeof(float) * num_threads);

    // Launch kernel
    compute_average<<<num_blocks, THREADS_PER_BLOCK>>>(d_results, num_threads);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_results, d_results, sizeof(float) * num_threads, cudaMemcpyDeviceToHost);

    // Compute the overall average on the host
    float final_sum = 0.0f;
    for (int i = 0; i < num_threads; i++) {
        final_sum += h_results[i];
    }

    float final_average = final_sum / num_threads;

    printf("Final average value: %f\n", final_average);
    printf("Expected theoretical value (a + c): %f\n", h_a + h_c);

    // Cleanup
    free(h_results);
    cudaFree(d_results);

    return 0;
}
