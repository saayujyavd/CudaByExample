#include <stdio.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>

#define N	100

__global__ void add(int* a, int* b, int* c)
{
	int tid = blockIdx.x;

	if(tid < N)
		c[tid] = a[tid] + b[tid++];
}

int main()
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;

	for(int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	
	cudaMalloc((void**)&dev_a, sizeof(int) * N);
	cudaMalloc((void**)&dev_b, sizeof(int) * N);
	cudaMalloc((void**)&dev_c, sizeof(int) * N);

	cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);

	add << <N, 1 >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i)
		printf("%2d + %2d = %2d\n", a[i], b[i], c[i]);

	cudaFree(dev_c);
	cudaFree(dev_b);
	cudaFree(dev_a);
	return(0);
}
