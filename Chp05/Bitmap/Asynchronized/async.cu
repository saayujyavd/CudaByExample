#include <cpu_bitmap.h>
#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>

#define DIM 1024
#define PI 3.1415926535897932f

__global__ void kernel(unsigned char* ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.x + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	__shared__ float cache[16][16];
	const float period = 128.0f;

	cache[threadIdx.x][threadIdx.y] =
		255.0f * (sinf(x * 2.0f * PI / period) + 1.0f) *
		(sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;

	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = 
		cache[blockDim.x - threadIdx.x - 1][blockDim.y - threadIdx.y - 1];

	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_ptr;

	size_t bitmap_size = bitmap.image_size();
	cudaMalloc((void**)&dev_ptr, bitmap_size);

	dim3 grid(DIM / 16, DIM / 16);
	dim3 threads_per_block(16, 16);
	kernel << <grid, threads_per_block >> > (dev_ptr);

	cudaMemcpy(bitmap.get_ptr(), dev_ptr, bitmap_size, cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();

	cudaFree(dev_ptr);
	return(0);
}
