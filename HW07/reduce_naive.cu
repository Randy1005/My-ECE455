#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <iomanip>
#include <chrono>

#define BLOCK_DIM 256
#define N (1 << 24)  // 16,777,216 elements

// ---------- CUDA Error Checking ----------
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " — "
                  << cudaGetErrorString(err) << " (" << func << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ---------- Random Initialization ----------
std::vector<int> create_rand_vector(size_t n, int min_val = 0, int max_val = 100) {
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    std::vector<int> vec(n);
    for (size_t i = 0; i < n; ++i)
        vec[i] = dist(e);
    return vec;
}

// ---------- CPU Reference Reduction ----------
int reduce_host(const std::vector<int>& data) {
    long long sum = 0;
    for (auto v : data)
        sum += v;
    return static_cast<int>(sum);
}

// ---------- GPU Naive Atomic Reduction ----------
__global__ void reduce_atomic(const int* in, int* out, size_t sz) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sz)
        atomicAdd(out, in[idx]);
}

// ---------- GPU Launcher ----------
int reduce_cuda_atomic(const std::vector<int>& h_in, float& time_ms) {
    int *d_in, *d_out;
    int h_out = 0;

    checkCuda(cudaMalloc(&d_in, sizeof(int) * N));
    checkCuda(cudaMalloc(&d_out, sizeof(int)));
    checkCuda(cudaMemcpy(d_in, h_in.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(d_out, 0, sizeof(int)));

    dim3 threads(BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    checkCuda(cudaEventRecord(start));
    reduce_atomic<<<blocks, threads>>>(d_in, d_out, N);
    checkCuda(cudaEventRecord(stop));
    checkCuda(cudaEventSynchronize(stop));
    checkCuda(cudaEventElapsedTime(&time_ms, start, stop));

    checkCuda(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));

    return h_out;
}

// ---------- Main ----------
int main() {
    std::cout << "Running Naive Atomic Reduction (N = " << N << ")..." << std::endl;

    std::vector<int> h_in = create_rand_vector(N);

    // ---- CPU reference + timing ----
    std::cout << "Computing CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    int ref = reduce_host(h_in);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // ---- GPU atomic reduction ----
    float gpu_time = 0.0f;
    int gpu_result = reduce_cuda_atomic(h_in, gpu_time);

    // ---- Validation ----
    bool ok = (ref == gpu_result);

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "CPU Result: " << ref << "\nGPU Result: " << gpu_result
              << "\nDifference: " << (ref - gpu_result) << std::endl;

    if (ok)
        std::cout << "Validation PASSED.\n";
    else
        std::cout << "Validation FAILED.\n";

    std::cout << "\n--- Performance ---\n";
    std::cout << "CPU time: " << cpu_time_ms << " ms\n";
    std::cout << "GPU kernel time: " << gpu_time << " ms\n";
    std::cout << "Speedup (CPU / GPU): " << (cpu_time_ms / gpu_time) << "x\n";

    return 0;
}
