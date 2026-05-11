#ifndef MAIN_H
#define MAIN_H

#include <book.h>
#include <cuda/cuda_runtime.h>
#include <vector>

namespace MyCudaStreams
{
    /**
     * @brief Checks whether the device supports concurrent kernel/copy overlap.
     * @return true if overlap is supported, false otherwise
     */
    bool overlapSupport()
    {
        cudaDeviceProp prop;
        int which_dev;

        HANDLE_ERROR(cudaGetDevice(&which_dev));
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, which_dev));

        if (!prop.deviceOverlap)
        {
            printf("Device will not handle overlaps, so no"
                "speed up from streams\n");
            return(false);
        }
        else return(true);
    }

    /**
     * @brief Creates one CUDA stream per slot in a pre-sized vector.
     * @param cuda_streams  Pre-sized vector to populate with stream handles
     * @param streams       Number of streams to create
     */
    void initCudaStreams(std::vector<cudaStream_t>& cuda_streams,
        size_t streams)
    {
        for (int i = 0; i < streams; ++i)
        {
            HANDLE_ERROR(cudaStreamCreate(&cuda_streams[i]));
        }
    }

    /**
     * @brief Allocates device memory for each pointer slot in the vector.
     * @param gpu_buffers       Vector of device pointers to populate
     * @param buffers           Number of buffers to allocate
     * @param each_buffer_size  Size in bytes of each buffer
     */
    template<typename T>
    void allocMemOnGpu(std::vector<T>& gpu_buffers, size_t buffers,
        size_t each_buffer_size)
    {
        for (int i = 0; i < buffers; ++i)
        {
            HANDLE_ERROR(cudaMalloc((void**)&gpu_buffers[i],
                each_buffer_size));
        }
    }

    /**
     * @brief Allocates page-locked (pinned) host memory for each pointer slot.
     *
     * Pinned memory is required for asynchronous DMA transfers to/from device.
     *
     * @param cpu_buffers       Vector of host pointers to populate
     * @param buffers           Number of buffers to allocate
     * @param each_buffer_size  Size in bytes of each buffer
     */
    template<typename T>
    void allocPageLockedMem(std::vector<T>& cpu_buffers, size_t buffers,
        size_t each_buffer_size)
    {
        for (int i = 0; i < buffers; ++i)
        {
            HANDLE_ERROR(cudaHostAlloc((void**)&cpu_buffers[i],
                each_buffer_size, cudaHostAllocDefault));
        }
    }

    /**
     * @brief Asynchronously copies num_buffers_to_copy consecutive buffer slots.
     *
     * dst_stride and src_stride are element offsets applied to each buffer
     * pointer before the copy, used to select the correct MB-sized chunk
     * within a FULL_DATA_SIZE-long buffer.
     *
     * @param dst_buffers           Destination buffer vector
     * @param src_buffers           Source buffer vector
     * @param each_buffer_size      Size in bytes of each individual copy
     * @param kind                  Direction of transfer (host<->device)
     * @param cuda_stream           Stream to issue the copy on
     * @param dst_start_idx         Starting index into dst_buffers
     * @param src_start_idx         Starting index into src_buffers
     * @param num_buffers_to_copy   Number of consecutive slots to copy
     * @param dst_stride            Element offset into each destination buffer
     * @param src_stride            Element offset into each source buffer
     */
    template<typename T>
    void copyLockedMemAsync(std::vector<T>& dst_buffers,
        std::vector<T>& src_buffers, size_t each_buffer_size,
        cudaMemcpyKind kind, cudaStream_t& cuda_stream,
        int dst_start_idx, int src_start_idx, size_t num_buffers_to_copy,
        size_t dst_stride, size_t src_stride)
    {
        for (int i = 0; i < num_buffers_to_copy; ++i)
        {
            HANDLE_ERROR(cudaMemcpyAsync(
                dst_buffers[dst_start_idx++] + dst_stride,
                src_buffers[src_start_idx++] + src_stride,
                each_buffer_size, kind, cuda_stream));
        }
    }

    /**
     * @brief Blocks until every stream in the vector has completed.
     * @param streams  Vector of stream handles to synchronize
     */
    void synchronizeCudaStreams(std::vector<cudaStream_t> streams)
    {
        for (int i = 0; i < streams.size(); ++i)
            HANDLE_ERROR(cudaStreamSynchronize(streams[i]));
    }

    /**
     * @brief Frees all pinned host buffers in reverse order.
     * @param cpu_buffers  Vector of pinned host pointers to free
     */
    template<typename T>
    void freePageLockedMem(std::vector<T>& cpu_buffers)
    {
        for (int i = cpu_buffers.size() - 1; i >= 0; --i)
            HANDLE_ERROR(cudaFreeHost(cpu_buffers[i]));
    }

    /**
     * @brief Frees all device buffers in reverse order.
     * @param gpu_buffers  Vector of device pointers to free
     */
    template<typename T>
    void freeMemOnGpu(std::vector<T>& gpu_buffers)
    {
        for (int i = gpu_buffers.size() - 1; i >= 0; --i)
            HANDLE_ERROR(cudaFree(gpu_buffers[i]));
    }

    /**
     * @brief Destroys all CUDA streams in reverse order.
     * @param streams  Vector of stream handles to destroy
     */
    void destroyCudaStreams(std::vector<cudaStream_t>& streams)
    {
        for (int i = streams.size() - 1; i >= 0; --i)
            HANDLE_ERROR(cudaStreamDestroy(streams[i]));
    }

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

} /* namespace MyCudaStreams */

#endif
