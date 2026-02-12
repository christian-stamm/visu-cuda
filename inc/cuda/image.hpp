#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

namespace cuda {
namespace image {

    void yuv2rgb(
        const uint8_t* d_yPlane, const uint8_t* d_uvPlane, uchar3* d_out, uint2 img_size, bool swapUV,
        cudaStream_t stream = 0);

    void rgb2bgr(const uchar3* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream = 0);
    void rgb2rgba(const uchar3* d_in, uchar4* d_out, uint2 img_size, cudaStream_t stream = 0);
    void rgba2rgb(const uchar4* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream = 0);

    void normalize(
        const uchar3* d_in, float3* d_out, uint2 img_size, float3 mean = make_float3(0.0f, 0.0f, 0.0f),
        float3 std = make_float3(1.0f, 1.0f, 1.0f), cudaStream_t stream = 0);

    void resize(const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out, cudaStream_t stream = 0);

    void padding(
        const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out,
        float3 pad_value = make_float3(0.0f, 0.0f, 0.0f), cudaStream_t stream = 0);

} // namespace image
} // namespace cuda
