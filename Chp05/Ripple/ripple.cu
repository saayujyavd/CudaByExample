#include "ripple.h"

__global__ void kernel(unsigned char* ptr, int ticks)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	float fx = x - DIM / 2;
	float fy = y - DIM / 2;

	float radius = sqrtf(fx * fx + fy * fy);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(radius / 10.0f - ticks / 7.0f) / (radius / 10.0f + 1.0f));

	ptr[offset * 4 + 0] = grey;
	ptr[offset * 4 + 1] = grey;
	ptr[offset * 4 + 2] = grey;
	ptr[offset * 4 + 3] = 255;
}

void Ripple::generateFrame(DataBlock* block, int ticks)
{
	dim3 blocks(DIM / 16, DIM / 16), threads(16, 16);
	kernel << <blocks, threads >> > (block->dev_bitmap, ticks);
	cudaMemcpy(block->bitmap->get_ptr(), block->dev_bitmap, block->bitmap->image_size(), cudaMemcpyDeviceToHost);
}
