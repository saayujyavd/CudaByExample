#include <stdio.h>

#define N	10

void add(int* a, int* b, int* c)
{
	int tid = 0;	// imagine this is CPU 0, so we start at 0
	while (tid < N)
		c[tid] = a[tid] + b[tid++];
}

int main()
{
	int a[N], b[N], c[N];

	for(int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	add(a, b, c);

	for (int i = 0; i < N; ++i)
		printf("%2d + %2d = %2d\n", a[i], b[i], c[i]);
	return(0);
}
