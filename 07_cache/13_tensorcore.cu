#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define TILE_SIZE 128
#define R_FRAGS   2
#define C_FRAGS   (TILE_SIZE / 16)
#define PAD 8 // padding. バンク競合を防ぐ

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       float *d_a, float *d_b, float *d_c) {
  int offset_a_m = TILE_SIZE * blockIdx.x;
  int offset_b_n = TILE_SIZE * blockIdx.y;
  int i = threadIdx.x;
  int warp_id = threadIdx.x / 32;

  __shared__ half block_a[2][16][TILE_SIZE + PAD];
  __shared__ half block_b[2][16][TILE_SIZE + PAD];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[R_FRAGS][C_FRAGS];
  for (int r = 0; r < R_FRAGS; r++)
    for (int c = 0; c < C_FRAGS; c++)
      wmma::fill_fragment(acc[r][c], 0.0f);

  int a_row_group = i % (TILE_SIZE / 4);
  int a_col_group = i / (TILE_SIZE / 4);
  int b_row_group = i % 4;
  int b_col_group = i / 4;

  for(int step = 0; step < 16; step += 4) {
    int a_col = a_col_group + step;
    int vec_idx = ((0 + a_col) * dim_m + offset_a_m) / 4 + a_row_group;
    float4 vec = reinterpret_cast<const float4*>(d_a)[vec_idx];
    block_a[0][a_col][a_row_group * 4 + 0] = __float2half(vec.x);
    block_a[0][a_col][a_row_group * 4 + 1] = __float2half(vec.y);
    block_a[0][a_col][a_row_group * 4 + 2] = __float2half(vec.z);
    block_a[0][a_col][a_row_group * 4 + 3] = __float2half(vec.w);
  }
  for (int step = 0; step < TILE_SIZE; step += (TILE_SIZE / 4)) {
    int b_col = b_col_group + step;
    int vec_idx = ((offset_b_n + b_col) * dim_k + 0) / 4 + b_row_group;
    float4 vec = reinterpret_cast<const float4*>(d_b)[vec_idx];
    block_b[0][b_row_group * 4 + 0][b_col] = __float2half(vec.x);
    block_b[0][b_row_group * 4 + 1][b_col] = __float2half(vec.y);
    block_b[0][b_row_group * 4 + 2][b_col] = __float2half(vec.z);
    block_b[0][b_row_group * 4 + 3][b_col] = __float2half(vec.w);
  }
  __syncthreads();

  for (int k = 16; k < dim_k; k += 32) { 
    for(int step = 0; step < 16; step += 4) {
      int a_col = a_col_group + step;
      int vec_idx = ((k + a_col) * dim_m + offset_a_m) / 4 + a_row_group;
      float4 vec = reinterpret_cast<const float4*>(d_a)[vec_idx];
      block_a[1][a_col][a_row_group * 4 + 0] = __float2half(vec.x);
      block_a[1][a_col][a_row_group * 4 + 1] = __float2half(vec.y);
      block_a[1][a_col][a_row_group * 4 + 2] = __float2half(vec.z);
      block_a[1][a_col][a_row_group * 4 + 3] = __float2half(vec.w);
    }
    for (int step = 0; step < TILE_SIZE; step += (TILE_SIZE / 4)) {
      int b_col = b_col_group + step;
      int vec_idx = ((offset_b_n + b_col) * dim_k + k) / 4 + b_row_group;
      float4 vec = reinterpret_cast<const float4*>(d_b)[vec_idx];
      block_b[1][b_row_group * 4 + 0][b_col] = __float2half(vec.x);
      block_b[1][b_row_group * 4 + 1][b_col] = __float2half(vec.y);
      block_b[1][b_row_group * 4 + 2][b_col] = __float2half(vec.z);
      block_b[1][b_row_group * 4 + 3][b_col] = __float2half(vec.w);
    }
    for (int r = 0; r < R_FRAGS; r++) {
      int row_tile = warp_id * R_FRAGS + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[0][0][row_tile * 16], TILE_SIZE + PAD);
      for (int c = 0; c < C_FRAGS; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[0][0][c * 16], TILE_SIZE + PAD);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
    __syncthreads();

    if (k + 16 < dim_k) {
      for(int step = 0; step < 16; step += 4) {
        int a_col = a_col_group + step;
        int vec_idx = (((k + 16) + a_col) * dim_m + offset_a_m) / 4 + a_row_group;
        float4 vec = reinterpret_cast<const float4*>(d_a)[vec_idx];
        block_a[0][a_col][a_row_group * 4 + 0] = __float2half(vec.x);
        block_a[0][a_col][a_row_group * 4 + 1] = __float2half(vec.y);
        block_a[0][a_col][a_row_group * 4 + 2] = __float2half(vec.z);
        block_a[0][a_col][a_row_group * 4 + 3] = __float2half(vec.w);
      }
      for (int step = 0; step < TILE_SIZE; step += (TILE_SIZE / 4)) {
        int b_col = b_col_group + step;
        int vec_idx = ((offset_b_n + b_col) * dim_k + (k + 16)) / 4 + b_row_group;
        float4 vec = reinterpret_cast<const float4*>(d_b)[vec_idx];
        block_b[0][b_row_group * 4 + 0][b_col] = __float2half(vec.x);
        block_b[0][b_row_group * 4 + 1][b_col] = __float2half(vec.y);
        block_b[0][b_row_group * 4 + 2][b_col] = __float2half(vec.z);
        block_b[0][b_row_group * 4 + 3][b_col] = __float2half(vec.w);
      }
    }
    for (int r = 0; r < R_FRAGS; r++) {
      int row_tile = warp_id * R_FRAGS + r;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(a_frag, &block_a[1][0][row_tile * 16], TILE_SIZE + PAD);
      for (int c = 0; c < C_FRAGS; c++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, &block_b[1][0][c * 16], TILE_SIZE + PAD);
        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }
    __syncthreads();
  }

  for (int r = 0; r < R_FRAGS; r++) {
    for (int c = 0; c < C_FRAGS; c++) {
      int c_m = offset_a_m + (warp_id * R_FRAGS + r) * 16;
      int c_n = offset_b_n + c * 16;
      if (c_n < dim_n && c_m < dim_m)
        wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();
  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
		 A, CUDA_R_32F, m, B, CUDA_R_32F, k, &beta,
		 C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
  
  int tile = TILE_SIZE; 
  dim3 block = dim3(tile);
  dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m, n, k, A, B, C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cublasDestroy(cublas_handle);
}