#include <cpu_bitmap.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>
#include "julia.h"

using namespace Julia;

__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x, y = blockIdx.y;
	int offset = y * gridDim.x + x;

	int julia_val = julia(x, y);
	ptr[offset * 4 + 0] = 255 * julia_val;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_ptr;

	size_t bitmap_size = bitmap.image_size();
	cudaMalloc((void**)&dev_ptr, bitmap_size);

	dim3 grid(DIM, DIM);
	kernel << <grid, 1 >> > (dev_ptr);

	cudaMemcpy(bitmap.get_ptr(), dev_ptr, bitmap_size, cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	
	cudaFree(dev_ptr);
	return(0);
}
