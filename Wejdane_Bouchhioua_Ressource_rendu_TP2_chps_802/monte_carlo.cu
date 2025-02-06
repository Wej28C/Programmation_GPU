#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

__constant__ float a, b, c;

__global__ void computeAverage(float *d_output, int num_samples) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(1234, idx, 0, &state);

    float sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float z = curand_normal(&state);
        sum += a * z * z + b * z + c;
    }

    d_output[idx] = sum / num_samples;
}

int main() {
    int num_threads = 1024;
    int num_blocks = 1024;
    int num_samples = 200;
    float h_a = 1.0f, h_b = 2.0f, h_c = 3.0f;

    // Allocate memory
    float *d_output;
    cudaMalloc((void **)&d_output, num_threads * num_blocks * sizeof(float));

    // Copy constants to device
    cudaMemcpyToSymbol(a, &h_a, sizeof(float));
    cudaMemcpyToSymbol(b, &h_b, sizeof(float));
    cudaMemcpyToSymbol(c, &h_c, sizeof(float));

    // Launch kernel
    computeAverage<<<num_blocks, num_threads>>>(d_output, num_samples);

    // Copy results back to host
    float *h_output = (float *)malloc(num_threads * num_blocks * sizeof(float));
    cudaMemcpy(h_output, d_output, num_threads * num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute final average
    float final_sum = 0.0f;
    for (int i = 0; i < num_threads * num_blocks; i++) {
        final_sum += h_output[i];
    }
    float final_avg = final_sum / (num_threads * num_blocks);

    printf("Final average: %f\n", final_avg);

    // Cleanup
    free(h_output);
    cudaFree(d_output);

    return 0;
}