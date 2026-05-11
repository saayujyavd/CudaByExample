#include <cuda/cuda_runtime.h>
#include <book.h>

#define SIZE	10 * 1024 * 1024

float cudaMallocTest(size_t size, bool is_up)
{
	cudaEvent_t start, stop;
	int* a, * dev_a;
	float time_taken;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	a = (int*)malloc(size * sizeof(*a));
	HANDLE_NULL(a);

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; ++i)
	{
		if (is_up)
			HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a),
				cudaMemcpyHostToDevice));
		else
			HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a),
				cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time_taken, start, stop));
	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaFree(dev_a));

	free(a);
	return(time_taken);
}

float cudaHostAllocTest(size_t size, bool is_up)
{
	cudaEvent_t start, stop;
	int* a, * dev_a;
	float time_taken;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(*a),
		cudaHostAllocDefault));
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, size * sizeof(*dev_a)));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	for (int i = 0; i < 100; ++i)
	{
		if (is_up)
			HANDLE_ERROR(cudaMemcpy(dev_a, a, size * sizeof(*dev_a),
				cudaMemcpyHostToDevice));
		else
			HANDLE_ERROR(cudaMemcpy(a, dev_a, size * sizeof(*dev_a),
				cudaMemcpyDeviceToHost));
	}

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&time_taken, start, stop));
	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaFree(dev_a));
	HANDLE_ERROR(cudaFreeHost(a));

	return(time_taken);
}

int main()
{
	bool up = true;
	float gb = (float)100 * SIZE * sizeof(int) / 1024 / 1024 / 1024;

	float malloc_up_time = cudaMallocTest(SIZE, up);
	float malloc_down_time = cudaMallocTest(SIZE, !up);

	float host_alloc_up_time = cudaHostAllocTest(SIZE, up);
	float host_alloc_down_time = cudaHostAllocTest(SIZE, !up);

	printf("Up time using cudaMalloc\t: %3.1f ms\n", malloc_up_time);
	printf("\tGB/s during up copy\t: %3.1f GB/s\n", gb / (malloc_up_time / 1000));

	printf("\nDown time using cudaMalloc\t: %3.1f ms\n", malloc_down_time);
	printf("\tGB/s during up copy\t: %3.1f GB/s\n", gb / (malloc_down_time / 1000));

	printf("\nUp time using cudaHostAlloc\t: %3.1f ms\n", host_alloc_up_time);
	printf("\tGB/s during up copy\t: %3.1f GB/s\n", gb / (host_alloc_up_time / 1000));

	printf("\nDown time using cudaHostAlloc\t: %3.1f ms\n", host_alloc_down_time);
	printf("\tGB/s during up copy\t: %3.1f GB/s\n", gb / (host_alloc_down_time / 1000));
	return(0);
}
