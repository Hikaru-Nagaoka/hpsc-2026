#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < range) {
        bucket[i] = 0;
    }
}

// バケットへのカウントをGPU上で並列実行する
__global__ void count_bucket(int *key, int *bucket, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
    }
}

// キー配列の再構築をGPU上で実行する
__global__ void reconstruct_key(int *key, int *bucket, int range) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i=0, j=0; i<range; i++) {
            for (; bucket[i]>0; bucket[i]--) {
                key[j++] = i;
            }
        }
    }
}

int main() {
  int n = 50;
  int range = 5;
  
  int *key;
  cudaMallocManaged(&key, n * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // CPUとGPU間で同じバケット配列データを共有するため
  int *bucket;
  cudaMallocManaged(&bucket, range * sizeof(int));

  int M = 256; // 1ブロックあたりのスレッド数
  int numBlocks_range = (range + M - 1) / M;
  int numBlocks_n = (n + M - 1) / M;

  init_bucket<<<numBlocks_range, M>>>(bucket, range);
  cudaDeviceSynchronize();

  count_bucket<<<numBlocks_n, M>>>(key, bucket, n);
  cudaDeviceSynchronize();

  reconstruct_key<<<1, 1>>>(key, bucket, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);

  return 0;
}