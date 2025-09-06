#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_unrolled(float* A, float* B_T, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / N;
    int col = tid % N;
    float sum = 0;
    if (row < N && col < N) {
        for (int k = 0; k < N; k += 4) {
            sum += A[row * N + k + 0] * B_T[col * N + k + 0];
            sum += A[row * N + k + 1] * B_T[col * N + k + 1];
            sum += A[row * N + k + 2] * B_T[col * N + k + 2];
            sum += A[row * N + k + 3] * B_T[col * N + k + 3];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_B_T = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            h_A[i * N + j] = 1.0f;
            h_B[i * N + j] = 1.0f;
            h_B_T[j * N + i] = h_B[i * N + j];
        }

    float *d_A, *d_B_T, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B_T, size);
    cudaMalloc(&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_T, h_B_T, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    matmul_unrolled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B_T, d_C, N);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0][0] = %f\n", h_C[0]);

    cudaFree(d_A);
    cudaFree(d_B_T);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_B_T);
    free(h_C);
    return 0;
}

