#ifndef HEAT_TRANSFER_H
#define HEAT_TRANSFER_H

#include <book.h>
#include <cpu_anim.h>
#include <cuda/cuda_runtime.h>

#define DIMX			1920
#define DIMY			1080
#define RATE_CONSTANT	0.25f

__global__ void copyHeatersKernel(cudaTextureObject_t, float*);
__global__ void blendKernel(cudaTextureObject_t, float*);

cudaTextureObject_t tex_const_input = 0;
cudaTextureObject_t tex_input = 0, tex_output = 0;

namespace HeatTransfer
{
	struct DataBlock
	{
		CPUAnimBitmap* bitmap;
		unsigned char* output_bitmap;

		cudaArray_t arr_const_input;
		cudaArray_t arr_input, arr_output;

		float* dev_input, *dev_output;
		float* dev_const_input;

		float total_time;
		float frames;

		cudaEvent_t start, stop;
	};

	void myCudaBindTexture2D(cudaTextureObject_t* tex_obj, cudaArray_t arr)
	{
		cudaResourceDesc res_desc = {};

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = arr;

		cudaTextureDesc tex_desc = {};
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.addressMode[0] = cudaAddressModeClamp;
		tex_desc.addressMode[1] = cudaAddressModeClamp;
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.normalizedCoords = 0;

		HANDLE_ERROR(cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL));
	}

	void animGpu(DataBlock* d, int ticks)
	{
		HANDLE_ERROR(cudaEventRecord(d->start, 0));

		dim3 blocks((DIMX + 15) / 16, (DIMY + 15) / 16);
		dim3 threads(16, 16);
		
		CPUAnimBitmap* bitmap = d->bitmap;
		bool dst_out = true;

		for (int i = 0; i < 100; ++i)
		{
			float* src_ptr = dst_out ? d->dev_input : d->dev_output;
			float* dst_ptr = dst_out ? d->dev_output : d->dev_input;
			cudaArray_t src_arr = dst_out ? d->arr_input : d->arr_output;
			cudaTextureObject_t src_tex = dst_out ? tex_input : tex_output;

			copyHeatersKernel << <blocks, threads >> > (tex_const_input, src_ptr);
			HANDLE_ERROR(cudaMemcpy2DToArray(src_arr, 0, 0, src_ptr,
				DIMX * sizeof(float), DIMX * sizeof(float), DIMY,
				cudaMemcpyDeviceToDevice));

			blendKernel << <blocks, threads >> > (src_tex, dst_ptr);
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

		cudaFreeArray(d->arr_output);
		cudaFreeArray(d->arr_input);
		cudaFreeArray(d->arr_const_input);

		cudaFree(d->dev_output);
		cudaFree(d->dev_input);
		cudaFree(d->dev_const_input);

		HANDLE_ERROR(cudaEventDestroy(d->start));
		HANDLE_ERROR(cudaEventDestroy(d->stop));
	}

	void allocArray(cudaArray_t* arr)
	{
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
		HANDLE_ERROR(cudaMallocArray(arr, &channel_desc, DIMX, DIMY));
	}
}

#endif
