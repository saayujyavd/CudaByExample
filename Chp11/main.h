#ifndef MAIN_H
#define MAIN_H

#include <cuda/cuda_runtime.h>
#include <book.h>

namespace Main
{
    /**
     * @brief Creates start and stop CUDA events and records the start event.
     *
     * Call this once before the timed region begins.
     */
    cudaEvent_t start, stop;

    void myCudaEventCreateAndRecord()
    {
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    }

    /**
     * @brief Records the stop event and returns elapsed time in milliseconds.
     *
     * Also destroys both events. Call this once after the timed region ends.
     *
     * @return Elapsed time in milliseconds
     */
    float myCudaEventElapsedTime()
    {
        float time_taken;

        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        HANDLE_ERROR(cudaEventElapsedTime(&time_taken, start, stop));
        HANDLE_ERROR(cudaEventDestroy(stop));
        HANDLE_ERROR(cudaEventDestroy(start));

        return(time_taken);
    }
}

#endif
