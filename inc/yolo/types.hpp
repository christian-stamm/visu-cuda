#pragma once
#include "corekit/cuda/core.hpp"
#include "corekit/cuda/draw.hpp"
#include "corekit/cuda/image.hpp"
#include "corekit/types.hpp"
#include "corekit/utils/color.hpp"
#include "corekit/utils/filemgr.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>

namespace yolo {

using namespace corekit::types;
using namespace corekit::utils;
using namespace corekit::cuda;

constexpr uint NET_WIDTH     = 640;
constexpr uint NET_HEIGHT    = 640;
constexpr uint MAX_DETS      = 300;
constexpr uint NUM_CLS       = 80;
constexpr uint NUM_KPS       = 17;
constexpr uint NUM_BNS       = 18;
constexpr uint MASK_COEFFS   = 32;
constexpr uint MASK_WIDTH    = 160;
constexpr uint MASK_HEIGHT   = 160;
constexpr uint MAX_PRIMITVES = 1024;

struct Transform {
    uint2 imgsize;
    uint2 netsize;
    uint2 padding;
    float scaling;
};

struct Letterbox {
    const Image3U& in;
    Image3U&       out;
    Transform      tf;
    float          conf;
    uint           dets;
};

struct BoundingBox {
    uint2 pos;
    uint2 dim;
    uint  clsid;
    float score;
};

struct Keypoint {
    uint  colID;
    uint2 pos;
    bool  vis;
};

struct Bone {
    uint origin;
    uint target;
};

using Proto = float[MASK_WIDTH * MASK_HEIGHT];
using Coeff = float[MASK_COEFFS];

using Skeleton = Keypoint[NUM_KPS];
using Result   = std::vector<BoundingBox>;

struct Storage {

    static Storage load(const Path& cfgFile, cudaStream_t stream)
    {
        JsonMap config = File::loadJson(cfgFile);

        Storage mem;

        std::vector<std::string> classes(NUM_CLS);
        std::vector<Keypoint>    keypts(NUM_KPS);
        std::vector<Bone>        bones;
        std::vector<Color>       palette;

        for (auto& [key, name] : config.at("classes").items()) {
            uint classId     = std::stoi(key);
            classes[classId] = name.get<std::string>();
        }

        for (auto& keypt : config.at("keypoints")) {
            const uint kpt_id = keypt.at("kptid").get<uint>();
            const uint col_id = keypt.at("color").get<uint>();

            keypts[kpt_id].colID = ((float)(col_id) / 15.0) * NUM_CLS;
        }

        for (auto& bone : config.at("bones")) {
            const uint origin = bone.at("origin").get<uint>();
            const uint target = bone.at("target").get<uint>();

            bones.push_back({origin, target});
        }

        palette = Color::sample(classes.size());

        cudaMallocAsync(&mem.d_boxes, MAX_DETS * sizeof(BoundingBox), stream);
        cudaMallocAsync(&mem.d_skelets, MAX_DETS * sizeof(Skeleton), stream);
        cudaMallocAsync(&mem.d_proto, MAX_DETS * sizeof(Proto), stream);
        cudaMallocAsync(&mem.d_coeffs, MAX_DETS * sizeof(Coeff), stream);
        cudaMallocAsync(&mem.d_bones, bones.size() * sizeof(Bone), stream);
        cudaMallocAsync(&mem.d_palette, classes.size() * sizeof(Color), stream);
        cudaMallocAsync(&mem.d_classes, classes.size() * MAX_TEXT_LEN, stream);

        cudaMallocAsync(&mem.d_rects, MAX_PRIMITVES * sizeof(Rect), stream);
        cudaMallocAsync(&mem.d_texts, MAX_PRIMITVES * sizeof(Text), stream);
        cudaMallocAsync(&mem.d_lines, MAX_PRIMITVES * sizeof(Line), stream);
        cudaMallocAsync(&mem.d_circles, MAX_PRIMITVES * sizeof(Circle), stream);

        mem.d_txtfont = Font::loadFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", DEF_TEXT_SIZE);

        cudaMemcpyAsync(mem.d_bones, bones.data(), bones.size() * sizeof(Bone), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(mem.d_palette, palette.data(), palette.size() * sizeof(Color), cudaMemcpyHostToDevice, stream);

        for (uint cls = 0; cls < classes.size(); cls++) {
            const Name&  name = classes.at(cls);
            char*        base = mem.d_classes + cls * MAX_TEXT_LEN;
            const size_t size = std::min<size_t>(name.size(), MAX_TEXT_LEN);
            cudaMemcpyAsync(base, name.c_str(), size * sizeof(char), cudaMemcpyHostToDevice, stream);
        }

        for (uint det = 0; det < MAX_DETS; det++) {
            cudaMemcpyAsync(&mem.d_skelets[det], keypts.data(), sizeof(Skeleton), cudaMemcpyHostToDevice, stream);
        }

        cudaStreamSynchronize(stream);
        return mem;
    };

    static void free(Storage& mem, cudaStream_t stream)
    {
        if (mem.d_boxes) {
            cudaFreeAsync(mem.d_boxes, stream);
            mem.d_boxes = nullptr;
        }

        if (mem.d_skelets) {
            cudaFreeAsync(mem.d_skelets, stream);
            mem.d_skelets = nullptr;
        }

        if (mem.d_proto) {
            cudaFreeAsync(mem.d_proto, stream);
            mem.d_proto = nullptr;
        }

        if (mem.d_coeffs) {
            cudaFreeAsync(mem.d_coeffs, stream);
            mem.d_coeffs = nullptr;
        }

        if (mem.d_bones) {
            cudaFreeAsync(mem.d_bones, stream);
            mem.d_bones = nullptr;
        }

        if (mem.d_palette) {
            cudaFreeAsync(mem.d_palette, stream);
            mem.d_palette = nullptr;
        }

        if (mem.d_classes) {
            cudaFreeAsync(mem.d_classes, stream);
            mem.d_classes = nullptr;
        }

        if (mem.d_rects) {
            cudaFreeAsync(mem.d_rects, stream);
            mem.d_rects = nullptr;
        }

        if (mem.d_texts) {
            cudaFreeAsync(mem.d_texts, stream);
            mem.d_texts = nullptr;
        }

        if (mem.d_lines) {
            cudaFreeAsync(mem.d_lines, stream);
            mem.d_lines = nullptr;
        }

        if (mem.d_circles) {
            cudaFreeAsync(mem.d_circles, stream);
            mem.d_circles = nullptr;
        }

        Font::freeFont(mem.d_txtfont);
    }

    BoundingBox* d_boxes   = nullptr;
    Skeleton*    d_skelets = nullptr;
    Bone*        d_bones   = nullptr;
    Proto*       d_proto   = nullptr;
    Coeff*       d_coeffs  = nullptr;
    Color*       d_palette = nullptr;
    char*        d_classes = nullptr;
    Font*        d_txtfont = nullptr;
    Line*        d_lines   = nullptr;
    Rect*        d_rects   = nullptr;
    Text*        d_texts   = nullptr;
    Circle*      d_circles = nullptr;
};

} // namespace yolo
