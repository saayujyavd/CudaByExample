#include <book.h>
#include <cuda/cuda_runtime.h>

#define SIZE	100 * 1024 * 1024	/* 100MB */

__global__ void histoKernel(unsigned char* buffer, size_t buffer_size, unsigned int* histo)
{
	int offset = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (offset < buffer_size)
	{
		atomicAdd(&(histo[buffer[offset]]), 1);
		offset += stride;
	}
}

int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	unsigned char* dev_buffer;
	unsigned int* dev_histo, histo[256];
	float time;

	HANDLE_ERROR(cudaMalloc((void**)&dev_buffer, SIZE));
	HANDLE_ERROR(cudaMalloc((void**)&dev_histo, 256 * sizeof(int)));

	HANDLE_ERROR(cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice));
	
	HANDLE_ERROR(cudaMemset(dev_histo, 0, 256 * sizeof(int)));

	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

	int blocks = prop.multiProcessorCount;
	histoKernel << <2 * blocks, 256 >> > (dev_buffer, SIZE, dev_histo);

	HANDLE_ERROR(cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&time, start, stop));

	printf("Time taken to generate: %3.1f ms\n", time);

	size_t histo_cnt = 0;
	for (int i = 0; i < 256; ++i)
		histo_cnt += histo[i];
	printf("Histo sum: %zd\n", histo_cnt);

	/* verify that we have the same GPU counts via CPU */
	for (int i = 0; i < SIZE; ++i)
		histo[(int)buffer[i]]--;

	for (int i = 0; i < 256; ++i)
		if (histo[i] != 0) printf("Failure at %d\n", i);

	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaEventDestroy(start));

	HANDLE_ERROR(cudaFree(dev_histo));
	HANDLE_ERROR(cudaFree(dev_buffer));

	free(buffer);
	return(0);
}
