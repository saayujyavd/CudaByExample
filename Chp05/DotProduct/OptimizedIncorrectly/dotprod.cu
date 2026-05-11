#include <stdio.h>
#include <stdlib.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>

#define imin(a, b) (a < b ? a : b)

const int N = 1024 * 33, threads_per_block = 256;

__global__ void dot(float* a, float* b, float* c)
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
	for (int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (cache_idx < i)
		{
			cache[cache_idx] += cache[cache_idx + i];
			__syncthreads();							// done for optimization, should hang the GPU when threads are not in multiples of a warp size
		}
	}
	if (cache_idx == 0) c[blockIdx.x] = cache[0];
}

int main()
{
	float* a, * b, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	const int blocks_per_grid = imin(32, (N + threads_per_block - 1) / threads_per_block);

	const size_t vec_size = sizeof(float) * N;
	const size_t partial_vec_size = sizeof(float) * blocks_per_grid;

	a = (float*)malloc(vec_size);
	b = (float*)malloc(vec_size);
	partial_c = (float*)malloc(partial_vec_size);

	cudaMalloc((void**)&dev_a, vec_size);
	cudaMalloc((void**)&dev_b, vec_size);
	cudaMalloc((void**)&dev_partial_c, partial_vec_size);

	for (int i = 0; i < N; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	cudaMemcpy(dev_a, a, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, vec_size, cudaMemcpyHostToDevice);

	dot << <blocks_per_grid, threads_per_block >> > (dev_a, dev_b, dev_partial_c);
	cudaMemcpy(partial_c, dev_partial_c, partial_vec_size, cudaMemcpyDeviceToHost);

	float sum = 0.0f;
	for (int i = 0; i < blocks_per_grid; ++i)
		sum += partial_c[i];

#define sumSquares(x) (x * (x + 1) * (2 * x + 1) / 6)
	printf("GPU value	= %.6g\nCorrect value	= %.6g\n", sum, 2.0f * sumSquares((float)(N - 1)));

	cudaFree(dev_partial_c);
	cudaFree(dev_b);
	cudaFree(dev_a);

	free(partial_c);
	free(b);
	free(a);

	return(0);
}
