#ifndef HEAT_TRANSFER_H
#define HEAT_TRANSFER_H

#include <book.h>
#include <cpu_anim.h>
#include <cuda/cuda_runtime.h>

#define DIM				1024
#define RATE_CONSTANT	0.25f

__global__ void copyHeatersKernel(float*, const float*);
__global__ void blendKernel(const float*, float*);

namespace HeatTransfer
{
	struct DataBlock
	{
		CPUAnimBitmap* bitmap;
		unsigned char* output_bitmap;

		float* dev_input, *dev_output;
		float* dev_const_input;

		float total_time;
		float frames;

		cudaEvent_t start, stop;
	};

	void animGpu(DataBlock* d, int ticks)
	{
		HANDLE_ERROR(cudaEventRecord(d->start, 0));

		dim3 blocks(DIM / 16, DIM / 16);
		dim3 threads(16, 16);
		
		CPUAnimBitmap* bitmap = d->bitmap;

		for (int i = 0; i < 500; ++i)
		{
			copyHeatersKernel << <blocks, threads >> > (d->dev_input, d->dev_const_input);
			blendKernel << <blocks, threads >> > (d->dev_input, d->dev_output);
			
			swap(d->dev_input, d->dev_output);
		}

		float_to_color << <blocks, threads >> > (d->output_bitmap, d->dev_input);
		HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), d->output_bitmap, bitmap->image_size(), cudaMemcpyDeviceToHost));

		HANDLE_ERROR(cudaEventRecord(d->stop, 0));
		HANDLE_ERROR(cudaEventSynchronize(d->stop));

		float time_elapsed;
		HANDLE_ERROR(cudaEventElapsedTime(&time_elapsed, d->start, d->stop));

		d->total_time += time_elapsed;
		d->frames++;
	}

	void animAndExit(DataBlock* d)
	{
		cudaFree(d->dev_input);
		cudaFree(d->dev_output);
		cudaFree(d->dev_const_input);

		HANDLE_ERROR(cudaEventDestroy(d->start));
		HANDLE_ERROR(cudaEventDestroy(d->stop));
	}
}

#endif
