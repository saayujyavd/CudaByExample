#include <cuda/device_launch_parameters.h>
#include <book.h>
#include <cuda/cuda_runtime.h>

namespace Main
{
	struct Lock
	{
		int* mutex;

		Lock()
		{
			int state = 0;

			HANDLE_ERROR(cudaMalloc((void**)&mutex, sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(mutex, &state, sizeof(int),
				cudaMemcpyHostToDevice));
		}

		~Lock()
		{
			cudaFree(mutex);
		}

		__device__ void lock()
		{
			while (true)
			{
				if (*mutex == 0)
				{
					*mutex = 1;
					break;
				}
			}
		}

		__device__ void unlock()
		{
			*mutex = 0;
		}
	};
}
