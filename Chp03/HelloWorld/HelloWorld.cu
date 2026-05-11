#include <stdio.h>

__global__ void add(int a, int b, int* c)
{
	*c = a + b;
}

int main(void)
{
	int c, * dev_c;

	cudaMalloc((void**)&dev_c, sizeof(int));	// from here onwards, dev_c cannot be used to read or write from mem. on host.
	add << <1, 1 >> > (2, 7, dev_c);			// also from now, dev_c cannot be dereferenced.

	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("2 + 7 = %d\n", c);
	cudaFree(dev_c);
	return(0);
}
