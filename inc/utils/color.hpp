#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace color {

struct RGBA {
    using List = std::vector<RGBA>;
    float r, g, b, a; // in [0,1]
};

// HSV (h in [0,360), s,v in [0,1]) to RGB (r,g,b in [0,1])
static RGBA hsv2rgba(float h, float s, float v, float a = 1.0f)
{
    h = std::fmod(std::fmod(h, 360.0f) + 360.0f, 360.0f); // wrap hue
    s = std::max(0.0f, std::min(1.0f, s));
    v = std::max(0.0f, std::min(1.0f, v));

    if (s == 0.0f) {
        return {v, v, v}; // gray
    }

    float c  = v * s;     // chroma
    float hh = h / 60.0f; // sector 0..6
    float x  = c * (1.0f - std::fabs(std::fmod(hh, 2.0f) - 1.0f));
    float r1 = 0, g1 = 0, b1 = 0;

    if (0.0f <= hh && hh < 1.0f) {
        r1 = c;
        g1 = x;
        b1 = 0;
    }
    else if (1.0f <= hh && hh < 2.0f) {
        r1 = x;
        g1 = c;
        b1 = 0;
    }
    else if (2.0f <= hh && hh < 3.0f) {
        r1 = 0;
        g1 = c;
        b1 = x;
    }
    else if (3.0f <= hh && hh < 4.0f) {
        r1 = 0;
        g1 = x;
        b1 = c;
    }
    else if (4.0f <= hh && hh < 5.0f) {
        r1 = x;
        g1 = 0;
        b1 = c;
    }
    else {
        r1 = c;
        g1 = 0;
        b1 = x;
    }

    float m = v - c;
    return {r1 + m, g1 + m, b1 + m, a};
}

// Generate N visually distinct colors using HSV space
static RGBA::List sample(int N)
{
    RGBA::List colors;
    for (int i = 0; i < N; ++i) {
        float h = (i * 360.0f) / N; // evenly spaced hues
        float s = 0.7f;             // fixed saturation
        float v = 0.9f;             // fixed value

        colors.push_back(hsv2rgba(h, s, v));
    }

    return colors;
}

}; // namespace color
