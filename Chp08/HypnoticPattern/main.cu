#define GL_GLEXT_PROTOTYPES

#include "Interop.h"

using namespace CudaGLInterop;

int main(int argc, char* argv[])
{
	void keyFunc(unsigned char, int, int);

	// set CUDA-GL interoperability
	setInterop(argc, argv);

	// launch our CUDA kernel
	launchCudaKernel();

	// set up GLUT and kick off main loop
	setGlutMainLoop(keyFunc);
	return(0);
}

__global__ void kernel(uchar4* ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x >= DIMX || y >= DIMY) return;
	int offset = y * DIMX + x;

	float fx = x / (float)DIMX - 0.5f;
	float fy = y / (float)DIMY - 0.5f;
	unsigned char green = 128 + 127 *
		sin(abs(fx * 100) - abs(fy * 100));

	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

void keyFunc(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		endProcess();
		break;
	default:
		break;
	}
}
