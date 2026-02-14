#pragma once

#include "yolo/base.hpp"
#include "yolo/types.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

namespace yolo {

using namespace corekit::cuda;

class Segm : public Base {

  public:
    using Ptr = std::shared_ptr<Segm>;

    using Base::Base;

    static uint buildSegm(
        const float* d_out,  // Output array of net [x0, y0, x1, y1, conf, class, 32 x coeff] * MAX_DETS
        Letterbox&   letbox, // Letterbox info for coordinate transformation
        Storage      mem,    // Memory manager for intermediate buffers
        cudaStream_t stream  // CUDA stream for asynchronous execution
    );

    static void drawSegm(
        Image3U&     d_img,  // Image data on device
        const float* d_mout, // mask protos
        Letterbox&   letbox, // Letterbox info for coordinate transformation
        Storage      mem,    // Memory manager containing boxes and masks
        uint         count,  // Number of valid boxes found
        cudaStream_t stream  // CUDA stream for asynchronous execution
    );

  protected:
    virtual void postprocess(Letterbox& letter) override
    {
        const IOBinding& b_box  = outputs.front();
        const IOBinding& b_msk  = outputs.back();
        const float*     d_mout = reinterpret_cast<float*>(b_box.ptr);
        const float*     d_pout = reinterpret_cast<float*>(b_msk.ptr);

        letter.dets = buildSegm(d_mout, letter, storage, stream);
        drawSegm(letter.out, d_pout, letter, storage, letter.dets, stream);
    }
};

} // namespace yolo
