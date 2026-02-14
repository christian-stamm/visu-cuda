#pragma once

#include "corekit/types.hpp"
#include "cuda/draw.hpp"
#include "cuda/image.hpp"
#include "cuda/model.hpp"
#include "utils/color.hpp"
#include "yolo/draw.hpp"
#include "yolo/types.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <string>
#include <vector>
#include <vector_types.h>

namespace yolo {

using namespace cuda::model;
using namespace cuda::image;

class Core : public Model {

  public:
    struct Settings {
        Path                     engine  = Path();
        std::vector<std::string> classes = {};
    };

    struct Letterbox {
        const uchar3* d_input;
        uchar3*       d_output;
        uint2         img_size;
        float         net_conf;
        float         img_scale;
        uint2         img_padding;
    };

    Core(const Settings& settings)
        : Model(settings.engine)
        , model_size(make_uint2(0, 0))
        , class_names(settings.classes)
        , num_clss(settings.classes.size())
        , max_dets(0)
    {
    }

    void process(const uchar3* d_input, uchar3* d_output, uint2 img_size, float net_conf = 0.25f)
    {
        Letterbox letter;
        letter.d_input  = d_input;
        letter.d_output = d_output;
        letter.img_size = img_size;
        letter.net_conf = net_conf;

        this->preprocess(letter);
        this->exec();
        this->postprocess(letter);
    }

  protected:
    void preprocess(Letterbox& letter)
    {
        const IOBinding& b_in = inputs.front();

        float ratioX   = (1.0f * model_size.x) / (1.0f * letter.img_size.x);
        float ratioY   = (1.0f * model_size.y) / (1.0f * letter.img_size.y);
        float mscale   = std::min<float>(ratioX, ratioY);
        uint  scaledW  = letter.img_size.x * mscale;
        uint  scaledH  = letter.img_size.y * mscale;
        uint2 rescaled = make_uint2(scaledW, scaledH);
        uint  paddingW = (model_size.x - scaledW) / 2;
        uint  paddingH = (model_size.y - scaledH) / 2;

        letter.img_scale   = mscale;
        letter.img_padding = make_uint2(paddingW, paddingH);

        resize(letter.d_input, cuda_resized, letter.img_size, rescaled, stream);
        padding(cuda_resized, cuda_padded, rescaled, model_size, make_uchar3(114, 144, 144), stream);
        uchar2float_nchw(cuda_padded, reinterpret_cast<float*>(b_in.ptr), model_size, stream);
    }

    void postprocess(Letterbox& letter)
    {
        const IOBinding& b_out  = outputs.front();
        const void*      d_mout = b_out.ptr;

        const size_t num_elems = b_out.size;
        const size_t num_dets  = num_elems / 6;
        const size_t num_pixel = letter.img_size.x * letter.img_size.y;

        if (num_elems < 6) {
            return;
        }

        std::vector<float> hostbuffer(num_elems);
        cudaMemcpyAsync(hostbuffer.data(), d_mout, hostbuffer.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaMemcpyAsync(letter.d_output, letter.d_input, num_pixel * sizeof(uchar3), cudaMemcpyDeviceToDevice, stream);

        std::vector<Rect> h_boxes;
        h_boxes.reserve(2 * num_dets);

        for (size_t i = 0; i < num_dets; i++) {
            float* det = &hostbuffer[i * 6];

            uint  l = det[5];
            float c = det[4];

            if (c < letter.net_conf) {
                continue;
            }

            uint x0 = (det[0] - letter.img_padding.x) / letter.img_scale;
            uint y0 = (det[1] - letter.img_padding.y) / letter.img_scale;
            uint x1 = (det[2] - letter.img_padding.x) / letter.img_scale;
            uint y1 = (det[3] - letter.img_padding.y) / letter.img_scale;

            x0 = std::max<uint>(0, std::min<uint>(x0, letter.img_size.x - 1));
            y0 = std::max<uint>(0, std::min<uint>(y0, letter.img_size.y - 1));
            x1 = std::max<uint>(0, std::min<uint>(x1, letter.img_size.x - 1));
            y1 = std::max<uint>(0, std::min<uint>(y1, letter.img_size.y - 1));

            uint cx = (x0 + x1) / 2;
            uint cy = (y0 + y1) / 2;
            uint w  = (x1 - x0);
            uint h  = (y1 - y0);

            Rect boxrect = {
                {cx, cy, w, h},
                {127, 0, 0, 127},
                {127, 0, 0, 50},
                2,
            };

            h_boxes.push_back(boxrect);
        }

        logger() << "Found " << 0.5f * h_boxes.size() << "Dets";

        if (h_boxes.empty()) {
            return;
        }

        Rect* d_boxes = nullptr;
        cudaMallocAsync(&d_boxes, h_boxes.size() * sizeof(Rect), stream);
        cudaMemcpyAsync(d_boxes, h_boxes.data(), h_boxes.size() * sizeof(Rect), cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);
        drawRect(letter.d_output, letter.img_size, d_boxes, h_boxes.size(), stream);

        cudaStreamSynchronize(stream);
        cudaFreeAsync(d_boxes, stream);
    }

    virtual bool prepare() override
    {
        bool success = Model::prepare();

        font = cuda::image::loadFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32);

        const IOBinding& b_in    = inputs.front();
        const IOBinding& b_out   = outputs.front();
        const size_t     height  = b_in.dims.d[2];
        const size_t     width   = b_in.dims.d[3];
        const size_t     numDets = b_out.dims.d[1];

        if (success) {
            this->model_size = make_uint2(width, height);
            this->max_dets   = numDets;

            cudaMallocAsync(&cuda_padded, width * height * sizeof(uchar3), stream);
            cudaMallocAsync(&cuda_resized, width * height * sizeof(uchar3), stream);

            if (!cuda_font && font.d_atlas && font.d_glyphs) {
                cudaMalloc(&cuda_font, sizeof(Font));
                cudaMemcpy(cuda_font, &font, sizeof(Font), cudaMemcpyHostToDevice);
            }
        }

        return true;
    }

    virtual bool cleanup() override
    {
        if (cuda_padded) {
            cudaFreeAsync(cuda_padded, stream);
            cudaFreeAsync(cuda_resized, stream);
            model_size   = make_uint2(0, 0);
            cuda_padded  = nullptr;
            cuda_resized = nullptr;
        }

        if (cuda_font) {
            cudaFree(cuda_font);
            cuda_font = nullptr;
        }

        if (font.d_atlas) {
            cudaFree(font.d_atlas);
            font.d_atlas = nullptr;
        }

        if (font.d_glyphs) {
            cudaFree(font.d_glyphs);
            font.d_glyphs = nullptr;
        }

        return Model::cleanup();
    }

    Font                     font;
    std::vector<std::string> class_names;

    uint2   model_size;
    uint    max_dets     = 0;
    uint    num_clss     = 0;
    uchar3* cuda_resized = nullptr;
    uchar3* cuda_padded  = nullptr;
    Font*   cuda_font    = nullptr;
};

} // namespace yolo
