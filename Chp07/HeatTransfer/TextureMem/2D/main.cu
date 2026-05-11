#include <cuda/device_launch_parameters.h>
#include "HeatTransfer.h"

using namespace HeatTransfer;

__global__ void copyHeatersKernel(cudaTextureObject_t tex_const_input, float* input)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	float output = tex2D<float>(tex_const_input, x, y);
	if (output > 0) input[offset] = output;
}

__global__ void blendKernel(cudaTextureObject_t src_tex, float* dst_ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= DIMX || y >= DIMY) return;
	int offset = y * DIMX + x;

	float c = tex2D<float>(src_tex, x, y);
	float l = tex2D<float>(src_tex, x - 1, y);
	float r = tex2D<float>(src_tex, x + 1, y);
	float t = tex2D<float>(src_tex, x, y + 1);
	float b = tex2D<float>(src_tex, x, y - 1);

	dst_ptr[offset] = c + RATE_CONSTANT * (l + r + t + b - 4 * c);
}

#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIMX, DIMY, &data);

	data.bitmap = &bitmap;
	data.total_time = 0;
	data.frames = 0;

	HANDLE_ERROR(cudaEventCreate(&data.start));
	HANDLE_ERROR(cudaEventCreate(&data.stop));

	size_t bitmap_size = bitmap.image_size();
	HANDLE_ERROR(cudaMalloc((void**)&data.output_bitmap, bitmap_size));

	HANDLE_ERROR(cudaMalloc((void**)&data.dev_input, bitmap_size));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_output, bitmap_size));
	HANDLE_ERROR(cudaMalloc((void**)&data.dev_const_input, bitmap_size));
	
	allocArray(&data.arr_const_input);
	allocArray(&data.arr_input);
	allocArray(&data.arr_output);

	float* temp = (float*)malloc(bitmap_size);
	for (int i = 0; i < DIMX * DIMY; ++i)
	{
		temp[i] = 0;

		int x = i % DIMX, y = i / DIMX;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}

	temp[DIMX * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
	temp[DIMX * 700 + 100] = MIN_TEMP;
	temp[DIMX * 300 + 300] = MIN_TEMP;
	temp[DIMX * 200 + 700] = MIN_TEMP;

	HANDLE_ERROR(cudaMemcpy2DToArray(data.arr_const_input, 0, 0, temp,
		sizeof(float) * DIMX, sizeof(float) * DIMX, DIMY, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy2DToArray(data.arr_input, 0, 0, temp,
		sizeof(float) * DIMX, sizeof(float) * DIMX, DIMY, cudaMemcpyHostToDevice));

	myCudaBindTexture2D(&tex_const_input, data.arr_const_input);
	myCudaBindTexture2D(&tex_input, data.arr_input);
	myCudaBindTexture2D(&tex_output, data.arr_output);

	for (int y = 800; y < 900; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIMX] = MIN_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_const_input, temp,
		bitmap_size, cudaMemcpyHostToDevice));
	
	for (int y = 800; y < DIMY; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIMX] = MAX_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_input, temp, 
		bitmap_size, cudaMemcpyHostToDevice));

	free(temp);
	bitmap.anim_and_exit((void (*)(void*, int))animGpu, 
		(void (*)(void*))animAndExit);

	return(0);
}
