#include <stdio.h>
#include <cuda/cuda_runtime.h>

int main()
{
	cudaDeviceProp dev;

	int devs;
	cudaGetDeviceCount(&devs);

	for (int i = 0; i < devs; ++i)
	{
		cudaGetDeviceProperties(&dev, i);

		printf("\n    --- General Information for device %d ---\n", i);
		printf("GPU Name:		%s\n", dev.name);
		printf("Compute capability:	%d.%d\n", dev.major, dev.minor);
		printf("Clock rate:		%d\n", dev.clockRate);

		printf("\n    --- Memory Information for device %d ---\n", i);
		printf("Total global mem:	%zd\n", dev.totalGlobalMem);
		
		printf("\n    --- Multiprocessor(MP) Information for device %d ---\n", i);
		printf("Multiprocessor count:	%d\n", dev.multiProcessorCount);
		printf("Shared mem per MP:	%zd\n", dev.sharedMemPerMultiprocessor);
		printf("Registers per MP:	%d\n", dev.regsPerBlock);
		printf("Threads in Warp:	%d\n", dev.warpSize);
		printf("Max threads per block:	%d\n", dev.maxThreadsPerBlock);
		printf("Max thread dimensions:	(%d, %d, %d)\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
		printf("Max grid dimension:	(%d, %d, %d)\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
	}
	return(0);
}
