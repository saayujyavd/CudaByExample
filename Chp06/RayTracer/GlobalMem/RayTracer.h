#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <cmath>
#include <cuda/cuda_runtime.h>

#undef INFINITY
#define INFINITY	2e10f

namespace RayTracer
{
	struct Point
	{
		float a, b, c;

		__device__ Point() {}
		__host__ __device__ Point(float a, float b, float c) : a(a), b(b), c(c) {}

		__device__ const Point operator-(const Point& p) const { return(Point(p.a - a, p.b - b, p.c - c)); }
		__device__ const Point operator*(float f) const { return(Point(a * f, b * f, c * f)); }
	};
	typedef Point Colour;

	struct Vec
	{
		Point a, b;

		__device__ Vec(Point a) : a(a), b(Point(0, 0, 0)) {}
		__device__ Vec(Point a, Point b) : a(a), b(b) {}

		__device__ const Vec operator-(const Point& a) const { return(Vec(b - a, Point(0, 0, 0))); }
		__device__ float length()
		{
			Vec v = *this - a;
			return(std::sqrt(v.a.a * v.a.a + v.a.b * v.a.b + v.a.c * v.a.c));
		}
	};

	struct Ray
	{
		Point orig, dest;
		float length;

		__device__ Ray(Point orig, Point dest) : orig(orig), dest(dest), length(Vec(orig - dest).length()) {}
	};

	struct Sphere
	{
		Point center;
		Colour colour;
		float radius;

		__device__ float hit(Ray from_pix, float* dist)
		{
			float x = from_pix.orig.a - center.a;
			float y = from_pix.orig.b - center.b;

			float x_square = x * x;
			float y_square = y * y;

			float r_square = radius * radius;
			if (x_square + y_square < r_square)
			{
				float dc = sqrtf(r_square - x_square - y_square);

				*dist = dc / radius;
				return(dc + center.c);
			}
			else return(-INFINITY);
		}
	};
}

#endif
