#pragma once

#include "yolo/types.hpp"

#include <cuda_runtime.h>

namespace yolo {

void buildBBox(
    const void* d_model_output, BoxResult* d_model_result, //
    uint2 model_size, uint2 img_size, float conf,          //
    const ColorPalette* cuda_palette,                      //
    const ClassList*    cuda_classes,                      //
    const Font*         cuda_font,                         //
    uint                max_dets,                          //
    bool                output_fp16,                       //
    cudaStream_t        stream = 0                         //
);

void drawBBox(
    uchar3* d_img, uint2 img_size,   //
    const BoxResult* d_model_result, //
    uint             max_dets,       //
    cudaStream_t     stream = 0      //
);

} // namespace yolo
