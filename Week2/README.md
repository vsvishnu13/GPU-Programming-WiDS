Week 2 — Parallel Thinking & CUDA Programming Basics
This week marks your transition from high-level GPU intuition to writing your first CUDA code.
You will learn how to map data-parallel problems onto threads, blocks, and grids, and how to think like a GPU programmer.
The goal is not to optimize yet—that begins in Week 3.
This week is about correctness, kernel structure, and understanding the CUDA execution model in practice.
Learning Goals
By the end of Week 2, you should be able to:
Write and launch simple CUDA kernels
Understand how threads compute indices (threadIdx, blockIdx, blockDim)
Map Python/Numpy-level operations into CUDA thread-level parallelism
Manage host–device memory transfers (basic cudaMemcpy)
Compile and run CUDA programs using nvcc or Colab’s CUDA environment
Choose grid/block dimensions appropriate for different workloads
This week provides the fundamental skills needed for memory optimization in Week 3.
Required Resources
1. NVIDIA CUDA Programming Guide — Chapters 3–4
These chapters introduce the CUDA execution model and the basic API.
Thread hierarchy
Kernels and launch configuration
Memory transfers and kernel arguments
Synchronization primitives
Link:
https://docs.nvidia.com/cuda/cuda-programming-guide/
3. CUDA Samples Repository
Browse and run the introductory examples:
vectorAdd
simpleKernel
matrixMul (just to inspect, not to implement yet)
GitHub:
https://github.com/NVIDIA/cuda-samples
You are not expected to understand every line — just get familiar with the structure.
4. Google Colab CUDA Setup (if using Colab)
Enable GPU:
Runtime → Change runtime type → GPU
Verify CUDA availability:
!nvidia-smi
!nvcc --version
Colab already includes CUDA 12.x toolchain.
Concepts Covered This Week
CUDA kernel syntax (__global__ functions)
Thread indexing and data parallelism
Grid/block configuration
Memory transfers (host ↔ device)
Kernel launches and synchronization
Basic debugging techniques (printf in kernels, bounds checks)
Starter Example
By the end of this week, this should make complete sense:
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
Week 2 Assignment (Summary)
Full details are in assignment.tex, but here is the assignment overview:
Task 1 — Implement Basic CUDA Kernels
Implement the following CUDA kernels:
Vector addition
Elementwise multiply-and-scale
ReLU activation (i.e., max(x, 0) per element)
For each kernel:
Write a complete .cu file
Allocate GPU memory
Copy input data to device
Launch the kernel
Copy results back to host
Compare with NumPy/Python results for correctness
You do not need to optimize; correctness only.
Task 2 — Grid/Block Design Exploration
Choose input sizes 
n
=
10
3
,
10
5
,
10
7
n=10 
3
 ,10 
5
 ,10 
7
  and experiment with:
blockDim.x = 32, 128, 256, 512
Appropriate gridDim.x
Record:
Total threads launched
How much of the work each thread performs
Which configurations result in correct output
This builds intuition for launch configurations and scalability.
Task 3 — (Optional) Profiling First Look
Use simple timers:
cudaEvent_t (preferred)
or
Python time.perf_counter() if embedding CUDA in PyTorch/CuPy
Record execution time for vector addition across different block sizes.
This sets the stage for serious profiling in Week 3.
Submission Folder
week2/
 ├── assignment.pdf
 ├── vector_add.cu
 ├── multiply_scale.cu
 ├── relu.cu
 ├── block_grid_experiments.md
 └── profiling_optional.md   # if attempted
Optional but Recommended
NVIDIA Blog: Even Easier Introduction to CUDA
Udacity: Intro to Parallel Programming — Lessons 1–2
Mark Harris: Optimizing Parallel Reduction in CUDA (preview for Week 3)
