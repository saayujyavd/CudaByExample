#include "main.h"

using namespace Main;

#define N	33 * 1024 * 1024
#define imin(a, b) (a < b ? a : b)

const int threads_per_block = 256;

__global__ void dot(size_t size, float* a, float* b, float* c)
{
	__shared__ float cache[threads_per_block];		// compiler automatically creates a copy of the shared variables for each block.
	int cache_idx = threadIdx.x;					// no need to incorporate block index because each block has its own pvt. copy.
	int tid = blockIdx.x * blockDim.x + cache_idx;	// just using cache_idx as threadIdx.x.

	float temp = 0;
	while (tid < size)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cache_idx] = temp;

	__syncthreads();
	for (int i = blockDim.x / 2; i > 0; i /= 2)
	{
		if (cache_idx < i) cache[cache_idx] += cache[cache_idx + i];
		__syncthreads();
	}
	if (cache_idx == 0) c[blockIdx.x] = cache[0];
}

float mallocTest(int size)
{
	float* a, * b, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;

	const int blocks_per_grid = imin(32, (size + threads_per_block - 1) / threads_per_block);
	const size_t vec_size = sizeof(float) * size;
	const size_t partial_vec_size = sizeof(float) * blocks_per_grid;

	a = (float*)malloc(vec_size);
	b = (float*)malloc(vec_size);
	partial_c = (float*)malloc(partial_vec_size);

	cudaMalloc((void**)&dev_a, vec_size);
	cudaMalloc((void**)&dev_b, vec_size);
	cudaMalloc((void**)&dev_partial_c, partial_vec_size);

	for (int i = 0; i < size; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	myCudaEventCreateAndRecord();

	cudaMemcpy(dev_a, a, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, vec_size, cudaMemcpyHostToDevice);

	dot << <blocks_per_grid, threads_per_block >> > (size, dev_a, dev_b, dev_partial_c);
	cudaMemcpy(partial_c, dev_partial_c, partial_vec_size, cudaMemcpyDeviceToHost);

	float time_taken = myCudaEventElapsedTime();
	float sum = 0.0f;

	for (int i = 0; i < blocks_per_grid; ++i)
		sum += partial_c[i];

	cudaFree(dev_partial_c);
	cudaFree(dev_b);
	cudaFree(dev_a);

	free(partial_c);
	free(b);
	free(a);

	printf("Value calculated: %f\n", sum);
	return(time_taken);
}

float myCudaHostAllocTest(int size)
{
	float* a, * b, * partial_c;
	float* dev_a, * dev_b, * dev_partial_c;
	const int blocks_per_grid = imin(32, (size + threads_per_block - 1) / threads_per_block);

	HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, 
		blocks_per_grid * sizeof(float),
		cudaHostAllocWriteCombined | cudaHostAllocMapped));

	for (int i = 0; i < size; ++i)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaHostGetDevicePointer((void**)&dev_a, a, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer((void**)&dev_b, b, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer((void**)&dev_partial_c, partial_c, 0));

	myCudaEventCreateAndRecord();

	dot << <blocks_per_grid, threads_per_block >> > (size, dev_a, dev_b, dev_partial_c);
	HANDLE_ERROR(cudaThreadSynchronize());

	float time_taken = myCudaEventElapsedTime();
	float sum = 0.0f;

	for (int i = 0; i < blocks_per_grid; ++i)
		sum += partial_c[i];

	cudaFreeHost(dev_partial_c);
	cudaFreeHost(dev_b);
	cudaFreeHost(dev_a);

	printf("Value calculated: %f\n", sum);
	return(time_taken);
}

int main(void)
{
	cudaDeviceProp prop;
	int dev;

	HANDLE_ERROR(cudaGetDevice(&dev));
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, dev));

	if (!prop.canMapHostMemory)
	{
		printf("Device can't map memory.\n");
		return(1);
	}
	else
		HANDLE_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));

	printf("Time using cudaMalloc: %3.1f ms\n", mallocTest(N));
	printf("Time using cudaHostAlloc: %3.1f ms\n", myCudaHostAllocTest(N));
	return(0);
}
