#ifndef HEAT_TRANSFER_H
#define HEAT_TRANSFER_H

#include <book.h>
#include <cpu_anim.h>
#include <cuda/cuda_runtime.h>

#define DIMX			1920
#define DIMY			1080
#define RATE_CONSTANT	0.25f

__global__ void copyHeatersKernel(cudaTextureObject_t, float*);
__global__ void blendKernel(cudaTextureObject_t, cudaTextureObject_t, float*, bool);

cudaTextureObject_t tex_const_input;
cudaTextureObject_t tex_input, tex_output;

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

		dim3 blocks(DIMX / 16, DIMY / 16);
		dim3 threads(16, 16);
		
		CPUAnimBitmap* bitmap = d->bitmap;

		volatile bool dst_out = true;
		for (int i = 0; i < 350; ++i)
		{
			float* in, * out;
			if (dst_out)
			{
				in = d->dev_input;
				out = d->dev_output;
			}
			else
			{
				in = d->dev_output;
				out = d->dev_input;
			}

			copyHeatersKernel << <blocks, threads >> > (tex_const_input, in);
			blendKernel << <blocks, threads >> > (tex_input, tex_output, out, dst_out);
			
			dst_out = !dst_out;
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
		cudaDestroyTextureObject(tex_const_input);
		cudaDestroyTextureObject(tex_input);
		cudaDestroyTextureObject(tex_output);

		cudaFree(d->dev_input);
		cudaFree(d->dev_output);
		cudaFree(d->dev_const_input);

		HANDLE_ERROR(cudaEventDestroy(d->start));
		HANDLE_ERROR(cudaEventDestroy(d->stop));
	}

	void myCudaBindTexture(cudaTextureObject_t* tex_obj, float* dev_ptr, size_t size)
	{
		cudaResourceDesc res_desc = {};

		res_desc.resType = cudaResourceTypeLinear;
		res_desc.res.linear.devPtr = dev_ptr;
		res_desc.res.linear.desc = cudaCreateChannelDesc<float>();
		res_desc.res.linear.sizeInBytes = size;

		cudaTextureDesc tex_desc = {};
		tex_desc.readMode = cudaReadModeElementType;

		HANDLE_ERROR(cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL));
	}
}

#endif
