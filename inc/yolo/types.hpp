#pragma once

#include "cuda/types.hpp"
#include "utils/color.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace yolo {

using namespace cuda::image;

struct BoundingBox {
    Rect  objbox;
    Rect  txtbox;
    Text  label;
    float score;
    uint  class_id;
};

struct ColorPalette {
    uint         count;
    color::RGBA* colors;

    static ColorPalette* build(const std::vector<std::string>& classNames)
    {
        const uint              count  = classNames.size();
        const color::RGBA::List colors = color::sample(count);

        // Allocate device memory for the color array
        color::RGBA* d_colors = nullptr;
        cudaMalloc(&d_colors, count * sizeof(color::RGBA));
        cudaMemcpy(d_colors, colors.data(), count * sizeof(color::RGBA), cudaMemcpyHostToDevice);

        // Build the struct on the host with device pointer
        ColorPalette h_palette;
        h_palette.count  = count;
        h_palette.colors = d_colors;

        // Allocate device memory for the struct and copy
        ColorPalette* d_palette = nullptr;
        cudaMalloc(&d_palette, sizeof(ColorPalette));
        cudaMemcpy(d_palette, &h_palette, sizeof(ColorPalette), cudaMemcpyHostToDevice);

        return d_palette;
    }

    static void release(ColorPalette* d_palette)
    {
        if (!d_palette) {
            return;
        }

        ColorPalette h_palette{};
        cudaMemcpy(&h_palette, d_palette, sizeof(ColorPalette), cudaMemcpyDeviceToHost);
        if (h_palette.colors) {
            cudaFree(h_palette.colors);
        }

        cudaFree(d_palette);
    }
};

struct ClassList {
    constexpr static uint MaxClasses = 1024;
    constexpr static uint MaxNameLen = 64;

    uint  count;
    char* names; // Pointer to device memory for class names

    static ClassList* build(const std::vector<std::string>& classNames)
    {
        if (classNames.size() > MaxClasses) {
            throw std::runtime_error("Cannot build classlist. Too many classes");
        }

        uint count = std::min<uint>(classNames.size(), MaxClasses);

        char hostNames[MaxClasses * MaxNameLen] = {};

        for (size_t i = 0; i < count; ++i) {
            const std::string& name = classNames[i];
            std::snprintf(hostNames + i * MaxNameLen, MaxNameLen, "%s", name.c_str());
        }

        // Allocate device memory for the names array
        char* d_names = nullptr;
        cudaMalloc(&d_names, MaxClasses * MaxNameLen * sizeof(char));
        cudaMemcpy(d_names, hostNames, MaxClasses * MaxNameLen * sizeof(char), cudaMemcpyHostToDevice);

        // Build the struct on the host with device pointer
        ClassList h_list;
        h_list.count = count;
        h_list.names = d_names;

        // Allocate device memory for the struct and copy
        ClassList* d_list = nullptr;
        cudaMalloc(&d_list, sizeof(ClassList));
        cudaMemcpy(d_list, &h_list, sizeof(ClassList), cudaMemcpyHostToDevice);

        return d_list;
    }

    static void release(ClassList* d_list)
    {
        if (!d_list) {
            return;
        }

        ClassList h_list{};
        cudaMemcpy(&h_list, d_list, sizeof(ClassList), cudaMemcpyDeviceToHost);
        if (h_list.names) {
            cudaFree(h_list.names);
        }
        cudaFree(d_list);
    }

    __device__ const char* getName(uint idx) const
    {
        if (idx >= count)
            return nullptr;
        return names + idx * MaxNameLen;
    }
};

struct BoxResult {
    uint         count;
    BoundingBox* bboxes;

    static BoxResult* build(uint maxDets)
    {
        BoxResult h_result;
        h_result.bboxes = nullptr;

        BoundingBox* d_bboxes = nullptr;
        cudaMalloc(&d_bboxes, maxDets * sizeof(BoundingBox));
        h_result.count  = 0;
        h_result.bboxes = d_bboxes;

        BoxResult* d_result = nullptr;
        cudaMalloc(&d_result, sizeof(BoxResult));
        cudaMemcpy(d_result, &h_result, sizeof(BoxResult), cudaMemcpyHostToDevice);

        return d_result;
    }

    static void release(BoxResult* d_result)
    {
        if (!d_result) {
            return;
        }

        BoxResult h_result{};
        cudaMemcpy(&h_result, d_result, sizeof(BoxResult), cudaMemcpyDeviceToHost);
        if (h_result.bboxes) {
            cudaFree(h_result.bboxes);
        }
        cudaFree(d_result);
    }
};

} // namespace yolo
