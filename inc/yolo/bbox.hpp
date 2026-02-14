#pragma once

#include "types.hpp"
#include "yolo/base.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

namespace yolo {

using namespace corekit::cuda;

class BBox : public Base {

  public:
    using Ptr = std::shared_ptr<BBox>;

    using Base::Base;

    static uint buildBBox(
        const float* d_out,   // Output array of net
        Letterbox&   letbox,  // Letterbox info for coordinate transformation
        Storage&     storage, // Memory manager for intermediate buffers
        cudaStream_t stream   // CUDA stream for asynchronous execution
    );

    static void drawBBox(
        Image3U&     d_img,   // Image data on device
        const uint   count,   // Number of valid boxes found
        Storage&     storage, // Memory manager for intermediate buffers
        cudaStream_t stream   // CUDA stream for asynchronous execution
    );

  protected:
    virtual void postprocess(Letterbox& letter) override
    {
        const IOBinding& b_out  = outputs.front();
        const float*     d_mout = reinterpret_cast<float*>(b_out.ptr);

        letter.dets = buildBBox(d_mout, letter, storage, stream);
        drawBBox(letter.out, letter.dets, storage, stream);
    }
};

} // namespace yolo
