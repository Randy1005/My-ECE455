#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_naive(float *A, float *B, float *C, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row = tid / N;
  int col = tid % N;
  if (row < N && col < N) {
    float sum = 0;
    for (int k = 0; k < N; ++k)
      sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

int main() {
  const int N = 1024;
  size_t size = N * N * sizeof(float);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  for (int i = 0; i < N * N; ++i) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

  matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  printf("C[0][0] = %f\n", h_C[0]);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}