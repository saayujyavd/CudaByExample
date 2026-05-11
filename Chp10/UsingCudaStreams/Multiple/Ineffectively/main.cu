#include <array>
#include "main.h"

using namespace MyCudaStreams;

#define MB              (1024 * 1024)
#define FULL_DATA_SIZE  (20 * MB)

/**
 * @brief Computes a blended 3-element running average over two input arrays.
 *
 * Each thread reads three consecutive elements (with 256-element wrap) from
 * both input arrays, averages each independently, then blends the two averages
 * into the output array.
 *
 * @param a  First input array (device)
 * @param b  Second input array (device)
 * @param c  Output array (device)
 */
__global__ void kernel(int* a, int* b, int* c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < MB - 2)
    {
        int idx1 = (idx + 1) % 256;
        int idx2 = (idx + 2) % 256;

        float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;

        c[idx] = (as + bs) / 2.0f;
    }
}

int main()
{
    // bail out early if the device does not support copy/kernel overlap
    if (!overlapSupport())
        return(0);

    // start the timer before any work begins
    myCudaEventCreateAndRecord();

    // each stream processes one MB-sized chunk concurrently
    const int streams = 25;
    std::vector<cudaStream_t> cuda_streams(streams);
    initCudaStreams(cuda_streams, streams);

    // host_buffers[0,1] = inputs a,b;  host_buffers[2] = output c
    // each host buffer is FULL_DATA_SIZE elements (pinned for async DMA)
    std::vector<int*>              host_buffers(3);

    // dev_buffers[j][0,1,2] = device a,b,c for stream j; each MB elements
    std::vector<std::vector<int*>> dev_buffers(streams,
        std::vector<int*>(3, nullptr));

    // allocate device memory — one set of 3 buffers per stream
    for (int i = 0; i < streams; ++i)
        allocMemOnGpu<int*>(dev_buffers[i], 3, MB * sizeof(int));

    // allocate pinned host memory — one buffer per slot, FULL_DATA_SIZE each
    allocPageLockedMem<int*>(host_buffers, host_buffers.size(),
        FULL_DATA_SIZE * sizeof(int));

    // fill input buffers with random data
    for (int i = 0; i < FULL_DATA_SIZE; ++i)
    {
        (host_buffers[0])[i] = rand();
        (host_buffers[1])[i] = rand();
    }

    // outer loop: each iteration hands one MB chunk to each stream
    // inner loop: stream j copies from host at element offset 'stride',
    //             runs the kernel, then copies the result back
    // stride guard: handles FULL_DATA_SIZE not divisible by streams
    for (int i = 0; i < streams * (FULL_DATA_SIZE / streams);
        i += MB * streams)
    {
        for (int j = 0, stride = i;
            (j < streams) && (stride < FULL_DATA_SIZE);
            ++j, stride += MB)
        {
            // H->D: copy input chunk at offset stride into stream j's device buffers
            copyLockedMemAsync<int*>(dev_buffers[j], host_buffers,
                MB * sizeof(int), cudaMemcpyHostToDevice, cuda_streams[j],
                0, 0, 2, 0, stride);

            // launch kernel on stream j's device buffers
            kernel << <MB / 256, 256, 0, cuda_streams[j] >> > (
                (dev_buffers[j])[0],
                (dev_buffers[j])[1],
                (dev_buffers[j])[2]);

            // D->H: copy result back into host output buffer at offset stride
            copyLockedMemAsync<int*>(host_buffers, dev_buffers[j],
                MB * sizeof(int), cudaMemcpyDeviceToHost, cuda_streams[j],
                2, 2, 1, stride, 0);
        }
    }

    // wait for all streams to finish, then report elapsed time
    synchronizeCudaStreams(cuda_streams);
    printf("Time taken: %3.1f ms\n", myCudaEventElapsedTime());

    // cleanup
    freePageLockedMem<int*>(host_buffers);

    for (int i = 0; i < streams; ++i)
        freeMemOnGpu<int*>(dev_buffers[i]);

    destroyCudaStreams(cuda_streams);
    return(0);
}
