#include "ripple.h"

using namespace Ripple;

int main()
{
	GPUAnimBitmap bitmap(DIMX, DIMY, NULL);
	bitmap.animAndExit(generateFrame, NULL);
}

__global__ void kernel(uchar4* ptr, int ticks)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= DIMX || y >= DIMY) return;
	int offset = y * DIMX + x;

	float fx = x - DIMX / 2;
	float fy = y - DIMY / 2;

	float radius = sqrtf(fx * fx + fy * fy);
	float wave = 128.0f + 127.0f * cosf(radius / 10.0f - ticks / 7.0f) / (radius / 10.0f + 1.0f);
	float t = wave / 255.0f;

	ptr[offset].x = (unsigned char)(t * t * 80.0f);
	ptr[offset].y = (unsigned char)(30.0f + t * 180.0f);
	ptr[offset].z = (unsigned char)(80.0f + t * 175.0f);
	ptr[offset].w = 255;
}

void Ripple::generateFrame(uchar4* pixels, void*, int ticks)
{
	dim3 grid((DIMX + 15) / 16, (DIMY + 15) / 16);
	dim3 threads(16, 16);
	kernel << <grid, threads >> > (pixels, ticks);
}
