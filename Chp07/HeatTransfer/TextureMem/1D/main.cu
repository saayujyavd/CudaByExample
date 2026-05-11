#include <cuda/device_launch_parameters.h>
#include "HeatTransfer.h"

using namespace HeatTransfer;

__global__ void copyHeatersKernel(cudaTextureObject_t tex_const_input, float* input)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	float output = tex1Dfetch<float>(tex_const_input, offset);
	if (output > 0) input[offset] = output;
}

__global__ void blendKernel(cudaTextureObject_t tex_input, cudaTextureObject_t tex_output, float* output, bool dst_out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	int left = (x == 0 ? offset : offset - 1);
	int right = (x == DIMX - 1 ? offset : offset + 1);
	int top = (y == 0 ? offset : offset - DIMX);
	int bottom = (y == DIMY - 1 ? offset : offset + DIMX);

	float c, l, r, t, b;
	if (dst_out)
	{
		c = tex1Dfetch<float>(tex_input, offset);
		l = tex1Dfetch<float>(tex_input, left);
		r = tex1Dfetch<float>(tex_input, right);
		t = tex1Dfetch<float>(tex_input, top);
		b = tex1Dfetch<float>(tex_input, bottom);
	}
	else
	{
		c = tex1Dfetch<float>(tex_output, offset);
		l = tex1Dfetch<float>(tex_output, left);
		r = tex1Dfetch<float>(tex_output, right);
		t = tex1Dfetch<float>(tex_output, top);
		b = tex1Dfetch<float>(tex_output, bottom);
	}

	output[offset] = c + RATE_CONSTANT * (l + r + t + b - 4 * c);
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

	myCudaBindTexture(&tex_const_input, data.dev_const_input, bitmap_size);
	myCudaBindTexture(&tex_input, data.dev_input, bitmap_size);
	myCudaBindTexture(&tex_output, data.dev_output, bitmap_size);

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

	for (int y = 800; y < 900; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIMX] = MIN_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_const_input, temp, bitmap_size, cudaMemcpyHostToDevice));
	
	for (int y = 800; y < DIMY; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIMX] = MAX_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_input, temp, bitmap_size, cudaMemcpyHostToDevice));

	free(temp);
	bitmap.anim_and_exit((void (*)(void*, int))animGpu, (void (*)(void*))animAndExit);

	return(0);
}
