#ifndef MAIN_H
#define MAIN_H

#include <book.h>
#include <cuda/cuda_runtime.h>

namespace MyCudaStreams
{
	bool overlapSupport()
	{
		cudaDeviceProp prop;
		int which_dev;

		HANDLE_ERROR(cudaGetDevice(&which_dev));
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, which_dev));

		if (!prop.deviceOverlap)
		{
			printf("Device will not handle overlaps, so no"
				"speed up from streams\n");
			return(false);
		}
		else return(true);
	}

	cudaEvent_t start, stop;

	void myCudaEventCreateAndRecord()
	{
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));
		HANDLE_ERROR(cudaEventRecord(start, 0));
	}

	float myCudaEventElapsedTime()
	{
		float time_taken;

		HANDLE_ERROR(cudaEventRecord(stop, 0));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		HANDLE_ERROR(cudaEventElapsedTime(&time_taken, start, stop));

		HANDLE_ERROR(cudaEventDestroy(stop));
		HANDLE_ERROR(cudaEventDestroy(start));
		return(time_taken);
	}
}

#endif
