#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>
#include "HeatTransfer.h"

using namespace HeatTransfer;

__global__ void copyHeatersKernel(float* input, const float* output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	if (output[offset] > 0) input[offset] = output[offset];
}

__global__ void blendKernel(const float* input, float* output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	int left = (x == 0 ? offset : offset - 1);
	int right = (x == DIM - 1 ? offset : offset + 1);
	int top = (y == 0 ? offset : offset - DIM);
	int bottom = (y == DIM - offset ? offset : offset + DIM);

	output[offset] = input[offset] + RATE_CONSTANT * (input[top] +
		input[bottom] + input[left] + input[right] - input[offset] * 4.0f);
}

#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f

int main()
{
	DataBlock data;
	CPUAnimBitmap bitmap(DIM, DIM, &data);

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

	float* temp = (float*)malloc(bitmap_size);
	for (int i = 0; i < DIM * DIM; ++i)
	{
		temp[i] = 0;

		int x = i % DIM, y = i / DIM;
		if ((x > 300) && (x < 600) && (y > 310) && (y < 601))
			temp[i] = MAX_TEMP;
	}

	temp[DIM * 100 + 100] = (MAX_TEMP + MIN_TEMP) / 2.0f;
	temp[DIM * 700 + 100] = MIN_TEMP;
	temp[DIM * 300 + 300] = MIN_TEMP;
	temp[DIM * 200 + 700] = MIN_TEMP;

	for (int y = 800; y < 900; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIM] = MIN_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_const_input, temp, bitmap_size, cudaMemcpyHostToDevice));
	
	for (int y = 800; y < DIM; ++y)
	{
		for (int x = 0; x < 200; ++x)
			temp[x + y * DIM] = MAX_TEMP;
	}
	HANDLE_ERROR(cudaMemcpy(data.dev_input, temp, bitmap_size, cudaMemcpyHostToDevice));

	free(temp);
	bitmap.anim_and_exit((void (*)(void*, int))animGpu, (void (*)(void*))animAndExit);

	return(0);
}
