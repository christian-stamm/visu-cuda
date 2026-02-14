#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

namespace cuda {
namespace image {

    struct GlyphInfo {
        int x;
        int y;
        int w;
        int h;
        int advance;
        int bearingX;
        int bearingY;
    };

    struct Font {
        int            atlasW;
        int            atlasH;
        int            glyphCount;
        int            ascent;
        int            descent;
        int            lineGap;
        unsigned char* d_atlas;  // device alpha atlas
        GlyphInfo*     d_glyphs; // device glyph metrics
    };

    struct Text {
        using Data = const char; // Pointer to text string in device memory

        const Font* font;
        Data*       msg;
        uint2       pos;   // (x, y) position of the text
        uchar4      color; // (r, g, b, a)
    };

    struct Rect {
        uint4  coords; // (cx, cy, w, h)
        uchar4 border; // (r, g, b, a)
        uchar4 fill;   // (r, g, b, a)
        uchar1 thickness;
    };

    struct Line {
        uint2  start;
        uint2  end;
        uchar4 color; // (r, g, b, a)
        uchar1 thickness;
    };

    struct Circle {
        uint2  center;
        uint1  radius;
        uchar4 border; // (r, g, b, a)
        uchar4 fill;   // (r, g, b, a)
        uchar1 thickness;
    };

} // namespace image
} // namespace cuda