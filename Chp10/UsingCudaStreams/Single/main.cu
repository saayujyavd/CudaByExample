#include "main.h"

using namespace MyCudaStreams;

#define MB				(1024 * 1024)
#define FULL_DATA_SIZE	(20 * MB)

__global__ void kernel(int* a, int* b, int* c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < MB)
	{
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;

		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

		c[idx] = (as + bs) / 2.0f;
	}
}

int main()
{
	myCudaEventCreateAndRecord();

	/* initialize a stream */
	cudaStream_t stream;
	HANDLE_ERROR(cudaStreamCreate(&stream));

	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	/* allocate mem on GPU */
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, MB * sizeof(*dev_a)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, MB * sizeof(*dev_b)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, MB * sizeof(*dev_c)));

	/* allocate page-locked mem, used to stream */
	HANDLE_ERROR(cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * 
		sizeof(*host_a), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * 
		sizeof(*host_b), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * 
		sizeof(*host_c), cudaHostAllocDefault));

	for (int i = 0; i < FULL_DATA_SIZE; ++i)
	{
		host_a[i] = rand();
		host_b[i] = rand();
	}

	/* now loop over full data, in bite-sized chunks */
	for (int i = 0; i < FULL_DATA_SIZE; i += MB)
	{
		// copy the page-locked mem to device, async
		HANDLE_ERROR(cudaMemcpyAsync(dev_a, host_a + i,
			MB * sizeof(int), cudaMemcpyHostToDevice, stream));
		HANDLE_ERROR(cudaMemcpyAsync(dev_b, host_a + i,
			MB * sizeof(int), cudaMemcpyHostToDevice, stream));
		
		kernel << <MB / 256, 256, 0, stream >> > (dev_a, dev_b, dev_c);

		// cpy data from dev back to page-locked mem
		HANDLE_ERROR(cudaMemcpyAsync(host_c + i, dev_c,
			MB * sizeof(int), cudaMemcpyDeviceToHost, stream));
	}
	/* copy result chunk from locked to full buffer */
	HANDLE_ERROR(cudaStreamSynchronize(stream));
	
	float time_taken = myCudaEventElapsedTime();

	HANDLE_ERROR(cudaFreeHost(host_c));
	HANDLE_ERROR(cudaFreeHost(host_b));
	HANDLE_ERROR(cudaFreeHost(host_a));

	HANDLE_ERROR(cudaFree(dev_c));
	HANDLE_ERROR(cudaFree(dev_b));
	HANDLE_ERROR(cudaFree(dev_a));

	printf("Time taken: %3.1f ms\n", time_taken);

	HANDLE_ERROR(cudaStreamDestroy(stream));
	return(0);
}
