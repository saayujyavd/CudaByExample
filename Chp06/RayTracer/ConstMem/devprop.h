#ifndef DEVPROP_H
#define DEVPROP_H

#include <stdio.h>
#include <cuda/cuda_runtime.h>

namespace Devprop
{
	void printDevNames(FILE* file)
	{
		cudaDeviceProp dev;

		int devs;
		cudaGetDeviceCount(&devs);

		for (int i = 0; i < devs; ++i)
		{
			cudaGetDeviceProperties(&dev, i);
			fprintf(file, "GPU Name:		%s\n", dev.name);
		}
	}
}

#endif
