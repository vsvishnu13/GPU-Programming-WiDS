#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// CUDA kernel: ReLU
__global__ void relu(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

int main() {
    int n = 1 << 20;              // ~1 million elements
    size_t size = n * sizeof(float);

    // Host memory
    float *h_x = new float[n];
    float *h_y = new float[n];

    // Initialize input (mix of positive & negative)
    for (int i = 0; i < n; i++) {
        h_x[i] = (i % 2 == 0) ? -1.0f : 2.0f;
    }

    // Device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy input to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    relu<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, n);

    // Error check + sync
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel error: "
                  << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // write GPU output to file
    FILE* f = fopen("output2.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%f\n", h_y[i]);   // or h_y[i]
    }
    fclose(f);

    // Verify
    std::cout << "h_y[0] = " << h_y[0] << std::endl; // expected 0
    std::cout << "h_y[5] = " << h_y[5] << std::endl; // expected 2

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;

    return 0;
}
