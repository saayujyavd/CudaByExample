#include <stdio.h>
#include <stdlib.h>
#include <cpu_bitmap.h>
#include <cuda/device_launch_parameters.h>
#include "RayTracer.h"

using namespace RayTracer;

#define rand(x) (x * rand() / RAND_MAX)

#define DIM		1024
#define SPHERES	20

__global__ void kernel(unsigned char* ptr, Sphere* spheres)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	Colour pix_col;
	Point pix = Point(x - DIM / 2.0f, y - DIM / 2.0f, 0.0f);
	float dist, temp, max_z = -INFINITY;

	for (int i = 0; i < SPHERES; ++i)
	{
		if ((temp = spheres[i].hit(Ray(pix, spheres[i].center), &dist)) > max_z)
		{
			pix_col = Colour(spheres[i].colour.a, spheres[i].colour.b, spheres[i].colour.c) * dist;
			max_z = temp;

			ptr[offset * 4 + 0] = (int)(pix_col.a * 255.0f);
			ptr[offset * 4 + 1] = (int)(pix_col.b * 255.0f);
			ptr[offset * 4 + 2] = (int)(pix_col.c * 255.0f);
			ptr[offset * 4 + 3] = 255;
		}
	}
}

int main()
{
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	Sphere* spheres;
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;

	size_t bitmap_size = bitmap.image_size();
	size_t spheres_size = sizeof(Sphere) * SPHERES;

	cudaMalloc((void**)&dev_bitmap, bitmap_size);
	cudaMalloc((void**)&spheres, spheres_size);

	Sphere* temp_s = (Sphere*)malloc(spheres_size);
	for (int i = 0; i < SPHERES; ++i)
	{
		temp_s[i].colour = Colour(rand(1.0f), rand(1.0f), rand(1.0f));
		temp_s[i].center = Point(rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f);
		temp_s[i].radius = rand(100.0f) + 20.0f;
	}

	cudaMemcpy(spheres, temp_s, spheres_size, cudaMemcpyHostToDevice);
	free(temp_s);
	
	dim3 grid(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel << <grid, threads >> > (dev_bitmap, spheres);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap_size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time_elapsed;
	cudaEventElapsedTime(&time_elapsed, start, stop);
	printf("Time taken to generate: %3.1f ms\n", time_elapsed);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	bitmap.display_and_exit();

	cudaFree(spheres);
	cudaFree(dev_bitmap);
	return(0);
}
