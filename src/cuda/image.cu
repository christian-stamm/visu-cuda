#include "cuda/image.hpp"

#include <cmath>
#include <cstdlib>

namespace cuda {
namespace image {

    /*
    #######################################################
    ##################### CUDA KERNELS ####################
    #######################################################
*/

    __device__ __forceinline__ uint8_t clamp_u8(int v, uint8_t lo = 0, uint8_t hi = 255)
    {
        return static_cast<uint8_t>(v < lo ? lo : (v > hi ? hi : v));
    }

    __device__ __forceinline__ int clamp_int(int v, int lo, int hi)
    {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    __global__ void nv16_to_rgb_kernel(
        const uint8_t* yPlane, const uint8_t* uvPlane, uchar3* out, int width, int height, bool swapUV)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const uint8_t* yrow  = yPlane + y * width;
        const uint8_t* uvrow = uvPlane + y * width;

        const int pair = x & ~1;
        const int Uidx = swapUV ? 1 : 0;
        const int Vidx = swapUV ? 0 : 1;

        const int U = uvrow[pair + Uidx];
        const int V = uvrow[pair + Vidx];
        const int Y = yrow[x];

        const float yf = static_cast<float>(Y);
        const float uf = static_cast<float>(U) - 128.0f;
        const float vf = static_cast<float>(V) - 128.0f;

        const float rf = yf + 1.402f * vf;
        const float gf = yf - 0.344136f * uf - 0.714136f * vf;
        const float bf = yf + 1.772f * uf;

        const int R = static_cast<int>(lrintf(rf));
        const int G = static_cast<int>(lrintf(gf));
        const int B = static_cast<int>(lrintf(bf));

        out[y * width + x] = make_uchar3(clamp_u8(R), clamp_u8(G), clamp_u8(B));
    }

    __global__ void rgb_to_bgr_kernel(const uchar3* in, uchar3* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx = y * width + x;
        const uchar3 rgb = in[idx];
        out[idx]         = make_uchar3(rgb.z, rgb.y, rgb.x);
    }

    __global__ void rgb_to_rgba_kernel(const uchar3* in, uchar4* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx = y * width + x;
        const uchar3 rgb = in[idx];
        out[idx]         = make_uchar4(rgb.x, rgb.y, rgb.z, 255);
    }

    __global__ void rgba_to_rgb_kernel(const uchar4* in, uchar3* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx  = y * width + x;
        const uchar4 rgba = in[idx];
        out[idx]          = make_uchar3(rgba.x, rgba.y, rgba.z);
    }

    __global__ void float2uint8_kernel(const float3* in, uchar3* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx = y * width + x;
        const float3 f   = in[idx];

        out[idx] = make_uchar3(clamp_u8(255 * f.x), clamp_u8(255 * f.y), clamp_u8(255 * f.z));
    }

    __global__ void uint82float_kernel(const uchar3* in, float3* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx = y * width + x;
        const uchar3 v   = in[idx];

        // Convert uint8 (0-255) to float (0-1) by dividing by 255
        out[idx] = make_float3(
            static_cast<float>(v.x) / 255.0f, static_cast<float>(v.y) / 255.0f, static_cast<float>(v.z) / 255.0f);
    }

    __global__ void uint82float_nchw_kernel(const uchar3* in, float* out, int width, int height)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) {
            return;
        }

        const int    idx = y * width + x;
        const uchar3 v   = in[idx];

        const int plane = width * height;
        const int r_idx = 0 * plane + idx;
        const int g_idx = 1 * plane + idx;
        const int b_idx = 2 * plane + idx;

        out[r_idx] = static_cast<float>(v.x) / 255.0f;
        out[g_idx] = static_cast<float>(v.y) / 255.0f;
        out[b_idx] = static_cast<float>(v.z) / 255.0f;
    }

    __global__ void resize_f3_kernel(
        const float3* in, int inW, int inH, float3* out, int outW, int outH, float scaleX, float scaleY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= outW || y >= outH) {
            return;
        }

        const float srcX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
        const float srcY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

        const int x0 = clamp_int(static_cast<int>(floorf(srcX)), 0, inW - 1);
        const int y0 = clamp_int(static_cast<int>(floorf(srcY)), 0, inH - 1);
        const int x1 = clamp_int(x0 + 1, 0, inW - 1);
        const int y1 = clamp_int(y0 + 1, 0, inH - 1);

        const float dx = srcX - static_cast<float>(x0);
        const float dy = srcY - static_cast<float>(y0);

        const int outIdx  = (y * outW + x);
        const int inIdx00 = (y0 * inW + x0);
        const int inIdx10 = (y0 * inW + x1);
        const int inIdx01 = (y1 * inW + x0);
        const int inIdx11 = (y1 * inW + x1);

        const float3 v00 = in[inIdx00];
        const float3 v10 = in[inIdx10];
        const float3 v01 = in[inIdx01];
        const float3 v11 = in[inIdx11];

        const float3 v0 =
            make_float3(v00.x + (v10.x - v00.x) * dx, v00.y + (v10.y - v00.y) * dx, v00.z + (v10.z - v00.z) * dx);

        const float3 v1 =
            make_float3(v01.x + (v11.x - v01.x) * dx, v01.y + (v11.y - v01.y) * dx, v01.z + (v11.z - v01.z) * dx);

        out[outIdx] = make_float3(v0.x + (v1.x - v0.x) * dy, v0.y + (v1.y - v0.y) * dy, v0.z + (v1.z - v0.z) * dy);
    }

    __global__ void resize_u3_kernel(
        const uchar3* in, int inW, int inH, uchar3* out, int outW, int outH, float scaleX, float scaleY)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= outW || y >= outH) {
            return;
        }

        const float srcX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
        const float srcY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

        const int x0 = clamp_int(static_cast<int>(floorf(srcX)), 0, inW - 1);
        const int y0 = clamp_int(static_cast<int>(floorf(srcY)), 0, inH - 1);
        const int x1 = clamp_int(x0 + 1, 0, inW - 1);
        const int y1 = clamp_int(y0 + 1, 0, inH - 1);

        const float dx = srcX - static_cast<float>(x0);
        const float dy = srcY - static_cast<float>(y0);

        const int outIdx  = (y * outW + x);
        const int inIdx00 = (y0 * inW + x0);
        const int inIdx10 = (y0 * inW + x1);
        const int inIdx01 = (y1 * inW + x0);
        const int inIdx11 = (y1 * inW + x1);

        const uchar3 v00 = in[inIdx00];
        const uchar3 v10 = in[inIdx10];
        const uchar3 v01 = in[inIdx01];
        const uchar3 v11 = in[inIdx11];

        const float3 v0 = make_float3(
            static_cast<float>(v00.x) + (static_cast<float>(v10.x) - static_cast<float>(v00.x)) * dx,
            static_cast<float>(v00.y) + (static_cast<float>(v10.y) - static_cast<float>(v00.y)) * dx,
            static_cast<float>(v00.z) + (static_cast<float>(v10.z) - static_cast<float>(v00.z)) * dx);

        const float3 v1 = make_float3(
            static_cast<float>(v01.x) + (static_cast<float>(v11.x) - static_cast<float>(v01.x)) * dx,
            static_cast<float>(v01.y) + (static_cast<float>(v11.y) - static_cast<float>(v01.y)) * dx,
            static_cast<float>(v01.z) + (static_cast<float>(v11.z) - static_cast<float>(v01.z)) * dx);

        const float3 v = make_float3(v0.x + (v1.x - v0.x) * dy, v0.y + (v1.y - v0.y) * dy, v0.z + (v1.z - v0.z) * dy);

        out[outIdx] = make_uchar3(
            clamp_u8(static_cast<int>(lrintf(v.x))), clamp_u8(static_cast<int>(lrintf(v.y))),
            clamp_u8(static_cast<int>(lrintf(v.z))));
    }

    __global__ void pad_f3_kernel(
        const float3* in, float3* out, int inW, int inH, int outW, int outH, int padLeft, int padTop, float3 padValue)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= outW || y >= outH) {
            return;
        }

        const int srcX = x - padLeft;
        const int srcY = y - padTop;

        const int outIdx = y * outW + x;

        if (srcX >= 0 && srcX < inW && srcY >= 0 && srcY < inH) {
            out[outIdx] = in[srcY * inW + srcX];
        }
        else {
            out[outIdx] = padValue;
        }
    }

    __global__ void pad_u3_kernel(
        const uchar3* in, uchar3* out, int inW, int inH, int outW, int outH, int padLeft, int padTop, uchar3 padValue)
    {
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= outW || y >= outH) {
            return;
        }

        const int srcX = x - padLeft;
        const int srcY = y - padTop;

        const int outIdx = y * outW + x;

        if (srcX >= 0 && srcX < inW && srcY >= 0 && srcY < inH) {
            out[outIdx] = in[srcY * inW + srcX];
        }
        else {
            out[outIdx] = padValue;
        }
    }

    /*
        #######################################################
        ##################### CPP WRAPPERS ####################
        #######################################################
    */

    void yuv2rgb(
        const uint8_t* d_yPlane, const uint8_t* d_uvPlane, uchar3* d_out, uint2 img_size, bool swapUV,
        cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        nv16_to_rgb_kernel<<<grid, block, 0, stream>>>(
            d_yPlane, d_uvPlane, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y), swapUV);
    }

    void rgb2bgr(const uchar3* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        rgb_to_bgr_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void rgb2rgba(const uchar3* d_in, uchar4* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        rgb_to_rgba_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void rgba2rgb(const uchar4* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        rgba_to_rgb_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void float2uchar(const float3* d_in, uchar3* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        float2uint8_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void uchar2float(const uchar3* d_in, float3* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        uint82float_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void uchar2float_nchw(const uchar3* d_in, float* d_out, uint2 img_size, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((img_size.x + block.x - 1) / block.x, (img_size.y + block.y - 1) / block.y);

        uint82float_nchw_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(img_size.x), static_cast<int>(img_size.y));
    }

    void resize(const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((size_out.x + block.x - 1) / block.x, (size_out.y + block.y - 1) / block.y);

        const float scaleX = static_cast<float>(size_in.x) / static_cast<float>(size_out.x);
        const float scaleY = static_cast<float>(size_in.y) / static_cast<float>(size_out.y);

        resize_f3_kernel<<<grid, block, 0, stream>>>(
            d_in, static_cast<int>(size_in.x), static_cast<int>(size_in.y), d_out, static_cast<int>(size_out.x),
            static_cast<int>(size_out.y), scaleX, scaleY);
    }

    void resize(const uchar3* d_in, uchar3* d_out, uint2 size_in, uint2 size_out, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((size_out.x + block.x - 1) / block.x, (size_out.y + block.y - 1) / block.y);

        const float scaleX = static_cast<float>(size_in.x) / static_cast<float>(size_out.x);
        const float scaleY = static_cast<float>(size_in.y) / static_cast<float>(size_out.y);

        resize_u3_kernel<<<grid, block, 0, stream>>>(
            d_in, static_cast<int>(size_in.x), static_cast<int>(size_in.y), d_out, static_cast<int>(size_out.x),
            static_cast<int>(size_out.y), scaleX, scaleY);
    }

    void padding(
        const float3* d_in, float3* d_out, uint2 size_in, uint2 size_out, float3 pad_value, cudaStream_t stream)
    {
        dim3 block(16, 16);
        dim3 grid((size_out.x + block.x - 1) / block.x, (size_out.y + block.y - 1) / block.y);

        const int padLeft = static_cast<int>(size_out.x - size_in.x) / 2;
        const int padTop  = static_cast<int>(size_out.y - size_in.y) / 2;

        pad_f3_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(size_in.x), static_cast<int>(size_in.y), static_cast<int>(size_out.x),
            static_cast<int>(size_out.y), padLeft, padTop, pad_value);
    }

    void padding(
        const uchar3* d_in, uchar3* d_out, uint2 size_in, uint2 size_out, uchar3 pad_value, cudaStream_t stream)
    {
        // d_out must be pre-allocated with enough space for the padded image (size_out.x * size_out.y)
        // No reallocation or temporary buffer is performed here.
        dim3 block(16, 16);
        dim3 grid((size_out.x + block.x - 1) / block.x, (size_out.y + block.y - 1) / block.y);

        const int padLeft = static_cast<int>(size_out.x - size_in.x) / 2;
        const int padTop  = static_cast<int>(size_out.y - size_in.y) / 2;

        pad_u3_kernel<<<grid, block, 0, stream>>>(
            d_in, d_out, static_cast<int>(size_in.x), static_cast<int>(size_in.y), static_cast<int>(size_out.x),
            static_cast<int>(size_out.y), padLeft, padTop, pad_value);
    }

} // namespace image
} // namespace cuda
