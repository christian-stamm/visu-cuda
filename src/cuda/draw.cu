#include "cuda/draw.hpp"

#include <cuda_runtime.h>
#include <ft2build.h>
#include FT_FREETYPE_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace cuda {
namespace image {

    namespace {
        constexpr int kFirstChar  = 32;
        constexpr int kLastChar   = 126;
        constexpr int kGlyphCount = (kLastChar - kFirstChar + 1);
        constexpr int kMaxTextLen = 256;

        struct GlyphInstance {
            const Font* font;
            int         glyphIndex;
            int         dstX;
            int         dstY;
            uchar4      color;
        };

        __device__ __forceinline__ uchar3 blend_rgb(uchar3 dst, uchar4 src, float alpha)
        {
            const float ia = 1.0f - alpha;
            return make_uchar3(
                static_cast<unsigned char>(dst.x * ia + src.x * alpha),
                static_cast<unsigned char>(dst.y * ia + src.y * alpha),
                static_cast<unsigned char>(dst.z * ia + src.z * alpha));
        }

        __device__ __forceinline__ float clampf(float v, float lo, float hi)
        {
            return v < lo ? lo : (v > hi ? hi : v);
        }

        inline bool is_device_pointer(const void* ptr)
        {
            cudaPointerAttributes attrs;
#if CUDART_VERSION >= 10000
            if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) {
                return false;
            }
            return attrs.type == cudaMemoryTypeDevice;
#else
            if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) {
                return false;
            }
            return attrs.memoryType == cudaMemoryTypeDevice;
#endif
        }

    } // namespace

    __global__ void draw_line_kernel(uchar3* img, int width, int height, Line line)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float x0 = static_cast<float>(line.start.x);
        const float y0 = static_cast<float>(line.start.y);
        const float x1 = static_cast<float>(line.end.x);
        const float y1 = static_cast<float>(line.end.y);

        const float px = static_cast<float>(x) + 0.5f;
        const float py = static_cast<float>(y) + 0.5f;

        const float dx   = x1 - x0;
        const float dy   = y1 - y0;
        const float len2 = dx * dx + dy * dy;

        float t = 0.0f;
        if (len2 > 0.0f) {
            t = ((px - x0) * dx + (py - y0) * dy) / len2;
        }
        t = clampf(t, 0.0f, 1.0f);

        const float cx = x0 + t * dx;
        const float cy = y0 + t * dy;

        const float dist2  = (px - cx) * (px - cx) + (py - cy) * (py - cy);
        const float radius = static_cast<float>(line.thickness.x) * 0.5f;

        if (dist2 <= radius * radius) {
            const int   idx = y * width + x;
            const float a   = static_cast<float>(line.color.w) / 255.0f;
            img[idx]        = blend_rgb(img[idx], line.color, a);
        }
    }

    __global__ void draw_circle_kernel(uchar3* img, int width, int height, Circle circle)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float cx = static_cast<float>(circle.center.x);
        const float cy = static_cast<float>(circle.center.y);
        const float r  = static_cast<float>(circle.radius.x);
        const float t  = static_cast<float>(circle.thickness.x);

        const float px = static_cast<float>(x) + 0.5f;
        const float py = static_cast<float>(y) + 0.5f;

        const float dx   = px - cx;
        const float dy   = py - cy;
        const float dist = sqrtf(dx * dx + dy * dy);

        const int idx = y * width + x;

        if (circle.fill.w > 0 && dist <= (r - t * 0.5f)) {
            const float a = static_cast<float>(circle.fill.w) / 255.0f;
            img[idx]      = blend_rgb(img[idx], circle.fill, a);
            return;
        }

        if (circle.border.w > 0 && fabsf(dist - r) <= t * 0.5f) {
            const float a = static_cast<float>(circle.border.w) / 255.0f;
            img[idx]      = blend_rgb(img[idx], circle.border, a);
        }
    }

    __global__ void draw_lines_kernel(uchar3* img, int width, int height, const Line* lines, int count)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float px = static_cast<float>(x) + 0.5f;
        const float py = static_cast<float>(y) + 0.5f;

        const int idx = y * width + x;
        uchar3    out = img[idx];

        for (int i = 0; i < count; ++i) {
            const Line  line   = lines[i];
            const float alpha  = static_cast<float>(line.color.w) / 255.0f;
            const float radius = static_cast<float>(line.thickness.x) * 0.5f;

            if (alpha <= 0.0f || radius <= 0.0f) {
                continue;
            }

            const float x0 = static_cast<float>(line.start.x);
            const float y0 = static_cast<float>(line.start.y);
            const float x1 = static_cast<float>(line.end.x);
            const float y1 = static_cast<float>(line.end.y);

            const float dx   = x1 - x0;
            const float dy   = y1 - y0;
            const float len2 = dx * dx + dy * dy;

            float t = 0.0f;
            if (len2 > 0.0f) {
                t = ((px - x0) * dx + (py - y0) * dy) / len2;
            }
            t = clampf(t, 0.0f, 1.0f);

            const float cx    = x0 + t * dx;
            const float cy    = y0 + t * dy;
            const float dist2 = (px - cx) * (px - cx) + (py - cy) * (py - cy);

            if (dist2 <= radius * radius) {
                out = blend_rgb(out, line.color, alpha);
            }
        }

        img[idx] = out;
    }

    __global__ void draw_rects_kernel(uchar3* img, int width, int height, const Rect* rects, int count)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float px = static_cast<float>(x) + 0.5f;
        const float py = static_cast<float>(y) + 0.5f;

        const int idx = y * width + x;
        uchar3    out = img[idx];

        for (int i = 0; i < count; ++i) {
            const Rect rect = rects[i];

            const float cx = static_cast<float>(rect.coords.x);
            const float cy = static_cast<float>(rect.coords.y);
            const float hw = static_cast<float>(rect.coords.z) * 0.5f;
            const float hh = static_cast<float>(rect.coords.w) * 0.5f;
            const float t  = static_cast<float>(rect.thickness.x);

            const float dx = fabsf(px - cx);
            const float dy = fabsf(py - cy);

            const float alphaFill   = static_cast<float>(rect.fill.w) / 255.0f;
            const float alphaBorder = static_cast<float>(rect.border.w) / 255.0f;

            float innerW = hw - t * 0.5f;
            float innerH = hh - t * 0.5f;
            if (innerW < 0.0f) {
                innerW = 0.0f;
            }
            if (innerH < 0.0f) {
                innerH = 0.0f;
            }

            if (alphaFill > 0.0f && dx <= innerW && dy <= innerH) {
                out = blend_rgb(out, rect.fill, alphaFill);
                continue;
            }

            if (alphaBorder > 0.0f && dx <= hw && dy <= hh && (dx > innerW || dy > innerH)) {
                out = blend_rgb(out, rect.border, alphaBorder);
            }
        }

        img[idx] = out;
    }

    __global__ void draw_circles_kernel(uchar3* img, int width, int height, const Circle* circles, int count)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const float px = static_cast<float>(x) + 0.5f;
        const float py = static_cast<float>(y) + 0.5f;

        const int idx = y * width + x;
        uchar3    out = img[idx];

        for (int i = 0; i < count; ++i) {
            const Circle circle = circles[i];

            const float cx = static_cast<float>(circle.center.x);
            const float cy = static_cast<float>(circle.center.y);
            const float r  = static_cast<float>(circle.radius.x);
            const float t  = static_cast<float>(circle.thickness.x);

            const float dx   = px - cx;
            const float dy   = py - cy;
            const float dist = sqrtf(dx * dx + dy * dy);

            const float alphaFill   = static_cast<float>(circle.fill.w) / 255.0f;
            const float alphaBorder = static_cast<float>(circle.border.w) / 255.0f;

            float innerR = r - t * 0.5f;
            if (innerR < 0.0f) {
                innerR = 0.0f;
            }

            if (alphaFill > 0.0f && dist <= innerR) {
                out = blend_rgb(out, circle.fill, alphaFill);
                continue;
            }

            if (alphaBorder > 0.0f && fabsf(dist - r) <= t * 0.5f) {
                out = blend_rgb(out, circle.border, alphaBorder);
            }
        }

        img[idx] = out;
    }

    __global__ void draw_glyph_kernel(
        uchar3* img, int imgW, int imgH, const unsigned char* atlas, int atlasW, GlyphInfo glyph, int dstX, int dstY,
        uchar4 color)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= glyph.w || y >= glyph.h) {
            return;
        }

        const int ix = dstX + x;
        const int iy = dstY + y;

        if (ix < 0 || ix >= imgW || iy < 0 || iy >= imgH) {
            return;
        }

        const int           atlasIdx = (glyph.y + y) * atlasW + (glyph.x + x);
        const unsigned char a8       = atlas[atlasIdx];
        if (a8 == 0) {
            return;
        }

        const float a   = (static_cast<float>(a8) / 255.0f) * (static_cast<float>(color.w) / 255.0f);
        const int   idx = iy * imgW + ix;
        img[idx]        = blend_rgb(img[idx], color, a);
    }

    __global__ void build_glyph_instances_kernel(const Text* texts, int count, GlyphInstance* instances)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }

        const Text&    txt = texts[idx];
        GlyphInstance* o   = instances + idx * kMaxTextLen;

        if (txt.font == nullptr || txt.font->d_atlas == nullptr || txt.font->d_glyphs == nullptr ||
            txt.msg == nullptr) {
            for (int i = 0; i < kMaxTextLen; ++i) {
                o[i].font       = nullptr;
                o[i].glyphIndex = -1;
            }
            return;
        }

        int penX     = static_cast<int>(txt.pos.x);
        int baseline = static_cast<int>(txt.pos.y) + txt.font->ascent;

        int i = 0;
        for (; i < kMaxTextLen; ++i) {
            const char c = txt.msg[i];
            if (c == '\0') {
                break;
            }

            const unsigned char uc = static_cast<unsigned char>(c);
            if (uc < kFirstChar || uc > kLastChar) {
                const GlyphInfo space = txt.font->d_glyphs[' ' - kFirstChar];
                penX += space.advance;
                o[i].font       = nullptr;
                o[i].glyphIndex = -1;
                continue;
            }

            const int        gidx = uc - kFirstChar;
            const GlyphInfo& g    = txt.font->d_glyphs[gidx];

            o[i].font       = txt.font;
            o[i].glyphIndex = gidx;
            o[i].dstX       = penX + g.bearingX;
            o[i].dstY       = baseline - g.bearingY;
            o[i].color      = txt.color;

            penX += g.advance;
        }

        for (; i < kMaxTextLen; ++i) {
            o[i].font       = nullptr;
            o[i].glyphIndex = -1;
        }
    }

    __global__ void draw_glyph_instances_kernel(
        uchar3* img, int imgW, int imgH, const GlyphInstance* instances, int totalInstances)
    {
        const int instanceIdx = blockIdx.x;
        if (instanceIdx >= totalInstances) {
            return;
        }

        const GlyphInstance inst = instances[instanceIdx];
        if (inst.font == nullptr || inst.glyphIndex < 0) {
            return;
        }

        const GlyphInfo g = inst.font->d_glyphs[inst.glyphIndex];
        if (g.w <= 0 || g.h <= 0) {
            return;
        }

        for (int y = threadIdx.y; y < g.h; y += blockDim.y) {
            for (int x = threadIdx.x; x < g.w; x += blockDim.x) {
                const int ix = inst.dstX + x;
                const int iy = inst.dstY + y;

                if (ix < 0 || ix >= imgW || iy < 0 || iy >= imgH) {
                    continue;
                }

                const int           atlasIdx = (g.y + y) * inst.font->atlasW + (g.x + x);
                const unsigned char a8       = inst.font->d_atlas[atlasIdx];
                if (a8 == 0) {
                    continue;
                }

                const float a   = (static_cast<float>(a8) / 255.0f) * (static_cast<float>(inst.color.w) / 255.0f);
                const int   idx = iy * imgW + ix;
                img[idx]        = blend_rgb(img[idx], inst.color, a);
            }
        }
    }

    Font loadFont(const char* fontPath, unsigned int size)
    {
        Font font{};
        if (fontPath == nullptr) {
            return font;
        }

        FT_Library ft   = nullptr;
        FT_Face    face = nullptr;
        if (FT_Init_FreeType(&ft) != 0) {
            return font;
        }
        if (FT_New_Face(ft, fontPath, 0, &face) != 0) {
            throw std::runtime_error("Failed to load font: " + std::string(fontPath));
        }

        const unsigned int pixelSize = size == 0 ? 32u : size;
        FT_Set_Pixel_Sizes(face, 0, pixelSize);

        font.ascent     = static_cast<int>(face->size->metrics.ascender / 64);
        font.descent    = static_cast<int>(-face->size->metrics.descender / 64);
        font.lineGap    = static_cast<int>((face->size->metrics.height / 64) - font.ascent - font.descent);
        font.glyphCount = kGlyphCount;

        int                    cellW = 0;
        int                    cellH = 0;
        std::vector<GlyphInfo> glyphs(kGlyphCount);

        for (int c = kFirstChar; c <= kLastChar; ++c) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
                continue;
            }
            const int          idx = c - kFirstChar;
            const FT_GlyphSlot g   = face->glyph;
            cellW                  = std::max(cellW, static_cast<int>(g->bitmap.width));
            cellH                  = std::max(cellH, static_cast<int>(g->bitmap.rows));

            glyphs[idx].w        = static_cast<int>(g->bitmap.width);
            glyphs[idx].h        = static_cast<int>(g->bitmap.rows);
            glyphs[idx].advance  = static_cast<int>(g->advance.x / 64);
            glyphs[idx].bearingX = static_cast<int>(g->bitmap_left);
            glyphs[idx].bearingY = static_cast<int>(g->bitmap_top);
        }

        if (cellW == 0 || cellH == 0) {
            FT_Done_Face(face);
            FT_Done_FreeType(ft);
            return font;
        }

        font.atlasW = cellW * kGlyphCount;
        font.atlasH = cellH;

        std::vector<unsigned char> atlas(static_cast<size_t>(font.atlasW) * static_cast<size_t>(font.atlasH), 0);

        for (int c = kFirstChar; c <= kLastChar; ++c) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER) != 0) {
                continue;
            }
            const int          idx     = c - kFirstChar;
            const FT_GlyphSlot g       = face->glyph;
            const int          offsetX = idx * cellW;
            const int          offsetY = 0;

            glyphs[idx].x = offsetX;
            glyphs[idx].y = offsetY;

            for (int row = 0; row < static_cast<int>(g->bitmap.rows); ++row) {
                const unsigned char* src = g->bitmap.buffer + row * g->bitmap.pitch;
                unsigned char*       dst = atlas.data() + (offsetY + row) * font.atlasW + offsetX;
                std::memcpy(dst, src, g->bitmap.width);
            }
        }

        cudaMalloc(&font.d_atlas, atlas.size());
        cudaMemcpy(font.d_atlas, atlas.data(), atlas.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&font.d_glyphs, sizeof(GlyphInfo) * glyphs.size());
        cudaMemcpy(font.d_glyphs, glyphs.data(), sizeof(GlyphInfo) * glyphs.size(), cudaMemcpyHostToDevice);

        FT_Done_Face(face);
        FT_Done_FreeType(ft);

        return font;
    }

    void drawText(uchar3* d_img, uint2 img_size, const Text* objs, int count, cudaStream_t stream)
    {
        if (d_img == nullptr || objs == nullptr) {
            return;
        }

        if (!is_device_pointer(objs)) {
            throw std::invalid_argument("Text objects must be in device memory");
        }

        const int totalInstances = count * kMaxTextLen;
        if (totalInstances <= 0) {
            return;
        }

        GlyphInstance* d_instances = nullptr;
        cudaMallocAsync(&d_instances, sizeof(GlyphInstance) * static_cast<size_t>(totalInstances), stream);

        const int threads = 128;
        const int blocks  = (count + threads - 1) / threads;
        build_glyph_instances_kernel<<<blocks, threads, 0, stream>>>(objs, count, d_instances);

        dim3 block(16, 16);
        dim3 grid(totalInstances, 1, 1);
        draw_glyph_instances_kernel<<<grid, block, 0, stream>>>(
            d_img, static_cast<int>(img_size.x), static_cast<int>(img_size.y), d_instances, totalInstances);

        cudaFreeAsync(d_instances, stream);
    }

    void drawLine(uchar3* d_img, uint2 img_size, const Line* objs, int count, cudaStream_t stream)
    {
        if (d_img == nullptr || objs == nullptr || count <= 0) {
            return;
        }

        if (!is_device_pointer(objs)) {
            throw std::invalid_argument("Line objects must be in device memory");
        }

        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        draw_lines_kernel<<<grid, block, 0, stream>>>(
            d_img, static_cast<int>(img_size.x), static_cast<int>(img_size.y), objs, count);
    }

    void drawRect(uchar3* d_img, uint2 img_size, const Rect* objs, int count, cudaStream_t stream)
    {
        if (d_img == nullptr || objs == nullptr || count <= 0) {
            return;
        }

        if (!is_device_pointer(objs)) {
            throw std::invalid_argument("Rect objects must be in device memory");
        }

        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        draw_rects_kernel<<<grid, block, 0, stream>>>(
            d_img, static_cast<int>(img_size.x), static_cast<int>(img_size.y), objs, count);
    }

    void drawCircle(uchar3* d_img, uint2 img_size, const Circle* objs, int count, cudaStream_t stream)
    {
        if (d_img == nullptr || objs == nullptr || count <= 0) {
            return;
        }

        if (!is_device_pointer(objs)) {
            throw std::invalid_argument("Circle objects must be in device memory");
        }

        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        draw_circles_kernel<<<grid, block, 0, stream>>>(
            d_img, static_cast<int>(img_size.x), static_cast<int>(img_size.y), objs, count);
    }

} // namespace image
} // namespace cuda
