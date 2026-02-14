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

    void float2uchar(const float3* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream = 0);
    void uchar2float(const uchar3* d_in, float3* d_out, uint2 img_size, cudaStream_t stream = 0);
    void uchar2float_nchw(const uchar3* d_in, float* d_out, uint2 img_size, cudaStream_t stream = 0);

    void resize(const uchar3* d_in, uchar3* d_out, uint2 size_in, uint2 size_out, cudaStream_t stream = 0);
    void resize(const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out, cudaStream_t stream = 0);

    // d_out must be pre-allocated with enough space for the padded image (size_out.x * size_out.y).
    // No reallocation or temporary buffer is performed inside this function.
    void padding(
        const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out, //
        float3 pad_value = make_float3(0.0f, 0.0f, 0.0f), cudaStream_t stream = 0);

    // d_out must be pre-allocated with enough space for the padded image (size_out.x * size_out.y).
    // No reallocation or temporary buffer is performed inside this function.
    void padding(
        const uchar3* d_in, uchar3* d_out, uint2 size_in, uint2 size_out, //
        uchar3 pad_value = make_uchar3(114, 114, 114), cudaStream_t stream = 0);

} // namespace image
} // namespace cuda
