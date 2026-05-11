#ifndef RIPPLE_H
#define RIPPLE_H

#include "Interop.h"
#include <cuda/device_launch_parameters.h>
#include <cuda/cuda_runtime.h>

using namespace CudaGLInterop;

namespace Ripple
{
	struct GPUAnimBitmap
	{
		GLuint buffer_obj;
		cudaGraphicsResource* resource;

		int width, height;
		int dragStartX, dragStartY;

		void* data_block;
		void (*fAnim)(uchar4*, void*, int);
		void (*animExit)(void*);
		void (*clickDrag)(void*, int, int, int, int);

		static GPUAnimBitmap** get_bitmap_ptr(void)
		{
			static GPUAnimBitmap* bitmap;
			return(&bitmap);
		}

		GPUAnimBitmap(int w, int h, void* d) : width(w), height(h), 
			data_block(d), clickDrag(NULL) 
		{
			*get_bitmap_ptr() = this;
			setInterop(&buffer_obj, &resource);
		}

		static void keyFunc(unsigned char key, int x, int y)
		{
			GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
			if (key == 27)   // Escape
			{
				if (bitmap->animExit)
					bitmap->animExit(bitmap->data_block);
				endProcess(bitmap->buffer_obj, bitmap->resource);
			}
		}

		static void idleFunc(void)
		{
			GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
			launchCudaKernel<GPUAnimBitmap>(bitmap);
		}

		void animAndExit(void (*f)(uchar4*, void*, int), void (*e)(void*))
		{
			fAnim = f;
			animExit = e;
			setGlutMainLoop(keyFunc, idleFunc);
		}
	};

	void generateFrame(uchar4*, void*, int);
}

#endif
