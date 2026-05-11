#include "main.h"
#include <cuda/cuda_runtime.h>
#include <book.h>
#include <ctime>

using namespace CpuHashTable;

#define SIZE		(100 * 1024 * 1024)
#define ELEMENTS	(SIZE / sizeof(unsigned int))
#define HASHENTRIES 1024

int main()
{
	unsigned int* buffer =
		(unsigned int*)big_random_block(SIZE);

	clock_t start, stop;
	start = clock();

	Table table;
	initTable(table, HASHENTRIES, ELEMENTS);

	for (int i = 0; i < ELEMENTS; ++i)
		addToTable(table, buffer[i], (void*)NULL);

	stop = clock();
	float time_taken = (float)(stop - start) /
		(float)CLOCKS_PER_SEC * 1000.0f;

	printf("Time to hash: %3.1f ms\n", time_taken);
	verifyTable(table, ELEMENTS);

	freeTable(table);
	free(buffer);
	return(0);
}
