#include <stdlib.h>
#include <book.h>
#include <cpu_bitmap.h>
#include <cuda/device_launch_parameters.h>
#include "RayTracer.h"
#include "devprop.h"

using namespace RayTracer;

#define rand(x) (x * rand() / RAND_MAX)

#define DIM		1024
#define SPHERES	20

__global__ void globalKernel(unsigned char* ptr, Sphere* spheres)
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

__constant__ Sphere dev_spheres[SPHERES];

__global__ void constKernel(unsigned char* ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * blockDim.x * gridDim.x + x;

	Colour pix_col;
	Point pix = Point(x - DIM / 2.0f, y - DIM / 2.0f, 0.0f);
	float dist, temp, max_z = -INFINITY;

	for (int i = 0; i < SPHERES; ++i)
	{
		if ((temp = dev_spheres[i].hit(Ray(pix, dev_spheres[i].center), &dist)) > max_z)
		{
			pix_col = Colour(dev_spheres[i].colour.a, dev_spheres[i].colour.b, dev_spheres[i].colour.c) * dist;
			max_z = temp;

			ptr[offset * 4 + 0] = (int)(pix_col.a * 255.0f);
			ptr[offset * 4 + 1] = (int)(pix_col.b * 255.0f);
			ptr[offset * 4 + 2] = (int)(pix_col.c * 255.0f);
			ptr[offset * 4 + 3] = 255;
		}
	}
}

FILE* file;

void renderUsingGlobalMem(CPUBitmap& bitmap)
{
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	Sphere* spheres;
	unsigned char* dev_bitmap;

	size_t bitmap_size = bitmap.image_size();
	size_t spheres_size = sizeof(Sphere) * SPHERES;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap_size));
	HANDLE_ERROR(cudaMalloc((void**)&spheres, spheres_size));

	Sphere* temp_s = (Sphere*)malloc(spheres_size);
	for (int i = 0; i < SPHERES; ++i)
	{
		temp_s[i].colour = Colour(rand(1.0f), rand(1.0f), rand(1.0f));
		temp_s[i].center = Point(rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f);
		temp_s[i].radius = rand(100.0f) + 20.0f;
	}

	HANDLE_ERROR(cudaMemcpy(spheres, temp_s, spheres_size, cudaMemcpyHostToDevice));
	free(temp_s);

	dim3 grid(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	globalKernel << <grid, threads >> > (dev_bitmap, spheres);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap_size, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));							// wait till all the GPU work completes

	float time_elapsed;
	HANDLE_ERROR(cudaEventElapsedTime(&time_elapsed, start, stop));
	fprintf(file, "Time taken to render using global mem:		%f ms\n", time_elapsed);

	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaEventDestroy(start));

	cudaFree(spheres);
	cudaFree(dev_bitmap);
}

void renderUsingConstMem(CPUBitmap& bitmap)
{
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));

	unsigned char* dev_bitmap;
	size_t bitmap_size = bitmap.image_size();
	size_t spheres_size = sizeof(Sphere) * SPHERES;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap_size));

	Sphere* temp_s = (Sphere*)malloc(spheres_size);
	for (int i = 0; i < SPHERES; ++i)
	{
		temp_s[i].colour = Colour(rand(1.0f), rand(1.0f), rand(1.0f));
		temp_s[i].center = Point(rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f, rand(1000.0f) - 500.0f);
		temp_s[i].radius = rand(100.0f) + 20.0f;
	}

	HANDLE_ERROR(cudaMemcpyToSymbol(dev_spheres, temp_s, spheres_size));	// used to copy from host mem to const mem on GPU
	free(temp_s);

	dim3 grid(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	constKernel << <grid, threads >> > (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap_size, cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));								// wait till all the GPU work completes

	float time_elapsed;
	HANDLE_ERROR(cudaEventElapsedTime(&time_elapsed, start, stop));
	fprintf(file, "Time taken to render using constant mem:	%f ms\n", time_elapsed);

	HANDLE_ERROR(cudaEventDestroy(stop));
	HANDLE_ERROR(cudaEventDestroy(start));

	cudaFree(dev_bitmap);
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);

	file = fopen("Benchmarks.txt", "a");
	HANDLE_NULL(file);

	Devprop::printDevNames(file);

	renderUsingGlobalMem(bitmap);
	renderUsingConstMem(bitmap);

	fprintf(file, "**************************************************************\n\n");
	fclose(file);

	bitmap.display_and_exit();
	return(0);
}
