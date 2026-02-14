#pragma once
#include "corekit/cuda/core.hpp"
#include "corekit/cuda/image.hpp"
#include "corekit/cuda/model.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace yolo {

using namespace corekit::cuda;

class Base : public Model {

  public:
    using Ptr = std::shared_ptr<Base>;

    struct Settings {
        Path engine;
        Path config;
    };

    Base(const Settings& settings)
        : Model(settings.engine)
        , config(settings.config)
        , netsize(make_uint2(NET_WIDTH, NET_HEIGHT))
    {
    }

    yolo::Result process(const Image3U& in, Image3U& out, float conf = 0.25f)
    {
        Transform tf = {
            .imgsize = in.getSize(),
            .netsize = netsize,
        };

        Letterbox letter = {
            .in   = in,
            .out  = out,
            .tf   = tf,
            .conf = conf,
            .dets = MAX_DETS,
        };

        cudaMemcpyAsync(out.data, in.data, in.getPixels() * sizeof(uchar3), cudaMemcpyDeviceToDevice, stream);

        this->preprocess(letter);
        this->exec();
        cudaStreamSynchronize(stream);
        this->postprocess(letter);

        yolo::Result result(letter.dets);
        cudaMemcpyAsync(
            result.data(), storage.d_boxes, letter.dets * sizeof(BoundingBox), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return result;
    }

  protected:
    void preprocess(Letterbox& letter)
    {
        const IOBinding& b_in = inputs.front();

        preprocess_into(letter.in, letter.tf, resized_cache, padded_cache, tensor_cache);

        cudaMemcpyAsync(
            b_in.ptr, tensor_cache.data, tensor_cache.getPixels() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    void preprocess_into(const Image3U& in, Transform& tf, Image3U& resized, Image3U& padded, Image1F& tensor)
    {
        const uint2 imgsize = in.getSize();

        const float scaledX = 1.0f * netsize.x / imgsize.x;
        const float scaledY = 1.0f * netsize.y / imgsize.y;
        const float scaling = std::min<float>(scaledX, scaledY);
        const uint  scaledW = (imgsize.x * scaling + 0.5f);
        const uint  scaledH = (imgsize.y * scaling + 0.5f);
        const uint  paddedW = (netsize.x - scaledW) / 2;
        const uint  paddedH = (netsize.y - scaledH) / 2;

        const uint2 scaled  = make_uint2(scaledW, scaledH);
        const uint2 padding = make_uint2(paddedW, paddedH);

        tf.imgsize = imgsize;
        tf.netsize = netsize;
        tf.scaling = scaling;
        tf.padding = padding;

        in.resize_into(resized, scaled);
        resized.pad_into(padded, netsize);
        padded.chnflip_into(tensor);
    }

    virtual void postprocess(Letterbox& letter) = 0;

    virtual bool prepare() override
    {
        bool success = Model::prepare();

        if (success) {
            storage = Storage::load(config, stream);
        }

        return true;
    }

    virtual bool cleanup() override
    {
        Storage::free(storage, stream);
        return Model::cleanup();
    }

    Path    config;
    Storage storage;
    uint2   netsize;
    Image3U resized_cache;
    Image3U padded_cache;
    Image1F tensor_cache;

}; // namespace yolo

} // namespace yolo
