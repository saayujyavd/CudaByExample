#include <stdio.h>
#include <stdlib.h>
#include "main.h"

using namespace Main;

#define imin(a, b) (a < b ? a : b)

const int N = 1024 * 1024 * 33, threads_per_block = 256;

__global__ void dot(Lock lock, float* a, float* b, float* c)
{
	__shared__ float cache[threads_per_block];		// compiler automatically creates a copy of the shared variables for each block.
	int cache_idx = threadIdx.x;					// no need to incorporate block index because each block has its own pvt. copy.
	int tid = blockIdx.x * blockDim.x + cache_idx;	// just using cache_idx as threadIdx.x.

	float temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cache_idx] = temp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cache_idx < i)
			cache[cache_idx] += cache[cache_idx + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cache_idx == 0)
	{
		lock.lock();
		*c += cache[0];
		lock.unlock();
	}
}

int main()
{
	float* a, * b, c = 0;
	float* dev_a, * dev_b, *dev_c;

	const int blocks_per_grid = imin(32, (N + threads_per_block - 1) / threads_per_block);
	const size_t vec_size = sizeof(float) * N;

	a = (float*)malloc(vec_size);
	b = (float*)malloc(vec_size);

	cudaMalloc((void**)&dev_a, vec_size);
	cudaMalloc((void**)&dev_b, vec_size);
	cudaMalloc((void**)&dev_c, sizeof(float));

	for (int i = 0; i < N; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, vec_size, cudaMemcpyHostToDevice);

	Lock lock;
	dot << <blocks_per_grid, threads_per_block >> > (lock, dev_a, dev_b, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

#define sumSquares(x) (x * (x + 1) * (2 * x + 1) / 6)
	printf("GPU value	= %.6g\nCorrect value	= %.6g\n", c, 2.0f * sumSquares((float)(N - 1)));

	lock.~Lock();

	cudaFree(dev_b);
	cudaFree(dev_a);
	cudaFree(dev_c);

	free(b);
	free(a);
	return(0);
}
