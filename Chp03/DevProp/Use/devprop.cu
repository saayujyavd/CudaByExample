#include <stdio.h>
#include <string.h>
#include <cuda/cuda_runtime.h>

int main()
{
	cudaDeviceProp prop;

	int dev;
	cudaGetDevice(&dev);
	printf("ID of current CUDA device:			%d\n", dev);

	memset(&prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;

	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to revision 1.3:	%d\n", dev);

	cudaSetDevice(dev);
	return(0);
}
