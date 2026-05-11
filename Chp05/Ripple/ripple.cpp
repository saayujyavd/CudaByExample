#include "ripple.h"

using namespace Ripple;

void cleanup(void* dev_ptr)
{
	cudaFree(dev_ptr);
}

int main()
{
	DataBlock data;

	CPUAnimBitmap bitmap(DIM, DIM, &data);
	data.bitmap = &bitmap;

	cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size());
	bitmap.anim_and_exit((void (*)(void*, int))generateFrame, (void (*)(void*))cleanup);
}
