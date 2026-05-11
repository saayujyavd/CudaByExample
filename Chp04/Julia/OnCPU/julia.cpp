#include <cpu_bitmap.h>
#include "julia.h"

using namespace Julia;

#define DIM	800

void kernel(unsigned char* ptr)
{
	for (int y = 0; y < DIM; ++y)
	{
		for (int x = 0; x < DIM; ++x)
		{
			int offset = y * DIM + x;

			int julia_val = julia(x, y);
			ptr[offset * 4 + 0] = 255 * julia_val;
			ptr[offset * 4 + 1] = 0;
			ptr[offset * 4 + 2] = 0;
			ptr[offset * 4 + 3] = 255;
		}
	}
}

int main()
{
	CPUBitmap bitmap(DIM, DIM);

	unsigned char* ptr = bitmap.get_ptr();
	kernel(ptr);

	bitmap.display_and_exit();
}
