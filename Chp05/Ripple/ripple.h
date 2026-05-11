#ifndef RIPPLE_H
#define RIPPLE_H

#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>
#include <cpu_anim.h>

#define DIM	800

namespace Ripple
{
	struct DataBlock
	{
		unsigned char* dev_bitmap;
		CPUAnimBitmap* bitmap;
	};

	void generateFrame(DataBlock*, int);
}

#endif
