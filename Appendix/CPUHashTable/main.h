#ifndef MAIN_H
#define MAIN_H

#include <cstdio>
#include <cstdlib>

namespace CpuHashTable
{
	struct Entry
	{
		unsigned int key;
		void* value;
		Entry* next;
	};

	struct Table
	{
		size_t cnt;
		Entry** entries;
		Entry* pool;
		Entry* first_free;
	};

	void initTable(Table& table, int entries,
		int elements)
	{
		table.cnt = entries;
		table.entries = (Entry**)calloc(entries,
			sizeof(Entry*));
		table.pool = (Entry*)malloc(elements 
			* sizeof(Entry));
		table.first_free = table.pool;
	}

	void freeTable(Table& table)
	{
		free(table.pool);
		free(table.entries);
	}

	size_t hash(unsigned int key, size_t cnt)
	{
		return(key % cnt);
	}

	void addToTable(Table& table, unsigned int key, void* value)
	{
		size_t hash_val = hash(key, table.cnt);
		Entry* location = table.first_free++;

		location->key = key;
		location->value = value;

		location->next = table.entries[hash_val];
		table.entries[hash_val] = location;
	}

	void verifyTable(const Table& table, size_t elements)
	{
		int cnt = 0;

		for (size_t i = 0; i < table.cnt; ++i)
		{
			Entry* current;
			while (current = table.entries[i])
			{
				if (hash((unsigned int)current->value, table.cnt) != i)
				{
					printf("%d hashed to %zd, but was"
						"located at %zd\n", (unsigned int)current->value,
						hash((unsigned int)current->value, table.cnt), i);
				}
				current = current->next;
			}
		}

		if (cnt != elements)
		{
			printf("%d elements found is hash table, should be %zd\n",
				cnt, elements);
		}
		else
			printf("All %d elements found in hash table.\n", cnt);
	}
}

#endif
