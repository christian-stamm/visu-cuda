#pragma once
#include "cuda/types.hpp"

#include <cuda_runtime.h>
#include <vector_types.h>

namespace cuda {
namespace image {

    Font loadFont(const char* fontPath, unsigned int size = 32);
    void drawText(uchar3* d_img, uint2 img_size, const Text* objs, int count, cudaStream_t stream = 0);
    void drawLine(uchar3* d_img, uint2 img_size, const Line* objs, int count, cudaStream_t stream = 0);
    void drawRect(uchar3* d_img, uint2 img_size, const Rect* objs, int count, cudaStream_t stream = 0);
    void drawCircle(uchar3* d_img, uint2 img_size, const Circle* objs, int count, cudaStream_t stream = 0);

} // namespace image
} // namespace cuda
