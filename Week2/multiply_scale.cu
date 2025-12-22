#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
// CUDA kernel
__global__ void mul_scale(const float* a,
                          const float* b,
                          float* c,
                          float alpha,
                          int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = alpha * a[i] * b[i];
    }
}

int main() {
    int n = 1 << 20;               // ~1 million elements
    size_t size = n * sizeof(float);
    float alpha = 2.0f;

    // Host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];

    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.5f;
        h_b[i] = 2.0f;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    mul_scale<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, alpha, n);

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch failed: "
                  << cudaGetErrorString(err) << std::endl;
    }
    // Synchrinization
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // write GPU output to file
    FILE* f = fopen("output1.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%f\n", h_c[i]);   // or h_y[i]
    }
    fclose(f);

    // Verify
    std::cout << "h_c[0] = " << h_c[0] << std::endl; // expected 6.0

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
