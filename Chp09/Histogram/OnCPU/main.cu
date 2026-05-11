#include <book.h>


#define SIZE	100 * 1024 * 1024	/* 100MB */


int main()
{
	unsigned char* buffer = (unsigned char*)big_random_block(SIZE);
	unsigned int* histo = (unsigned int*)calloc(256, sizeof(int));

	for (int i = 0; i < SIZE; ++i)
		histo[(int)buffer[i]]++;

	size_t histo_cnt = 0;
	for (int i = 0; i < 256; ++i)
		histo_cnt += histo[i];
	printf("Histo sum: %zd\n", histo_cnt);

	free(histo);
	free(buffer);
	return(0);
}
