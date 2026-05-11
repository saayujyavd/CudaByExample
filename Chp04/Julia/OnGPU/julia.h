#ifndef JULIA_H
#define JULIA_H

#define DIM	800

namespace Julia
{
	struct Complex
	{
		float r, i;

		__device__ Complex(float a, float b) : r(a), i(b) {}

		__device__ float mag_square(void) { return(r * r + i * i); }

		__device__ Complex operator*(const Complex& a) { return(Complex(r * a.r - i * a.i, i * a.r + r * a.i)); }
		__device__ Complex operator+(const Complex& a) { return(Complex(r + a.r, i + a.i)); }
	};

	__device__ int julia(int x, int y)
	{
		const float scale = 1.5f;

		int half_dim = DIM / 2;
		float jx = scale * (float)(half_dim - x) / half_dim;
		float jy = scale * (float)(half_dim - y) / half_dim;

		Complex c(-0.8f, 0.156f), zn(jx, jy);
		for (int n = 0; n < 200; ++n)
		{
			zn = zn * zn + c;
			if (zn.mag_square() > 1000) return(0);
		}
		return(1);
	}
}

#endif
