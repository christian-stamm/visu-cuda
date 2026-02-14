#pragma once

#include "yolo/base.hpp"
#include "yolo/types.hpp"

namespace yolo {

using namespace corekit::cuda;

class Pose : public Base {

  public:
    using Ptr = std::shared_ptr<Pose>;

    using Base::Base;

    static uint buildPose(
        const float* d_out,  // Output array of net
        Letterbox&   letbox, // Letterbox info for coordinate transformation
        Storage&     mem,    // Memory manager for intermediate buffers
        cudaStream_t stream  // CUDA stream for asynchronous execution
    );

    static void drawPose(
        Image3U&     d_img, // Image data on device
        const uint   count, // Number of valid boxes found
        Storage&     mem,   // Memory manager for intermediate buffers
        cudaStream_t stream // CUDA stream for asynchronous execution
    );

  protected:
    virtual void postprocess(Letterbox& letter) override
    {
        const IOBinding& b_out  = outputs.front();
        const float*     d_mout = reinterpret_cast<float*>(b_out.ptr);

        letter.dets = buildPose(d_mout, letter, storage, stream);
        drawPose(letter.out, letter.dets, storage, stream);
    }
};

} // namespace yolo
