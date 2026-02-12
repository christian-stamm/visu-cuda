#include "corekit/core.hpp"
#include "cuda/image.hpp"

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <string>
#include <vector_types.h>

using namespace corekit::types;
using namespace corekit::utils;

int main(int argc, char* argv[])
{
    const size_t width    = 1920;
    const size_t height   = 1080;
    const size_t numPixel = width * height;
    const uint2  imgSize  = make_uint2(width, height);

    std::string bytestream = File::loadTxt("/home/z0286456/Documents/cuda-visu/res/frame.yuv");

    uchar2* host_yuv_ptr = nullptr;
    void*   host_rgb_ptr = nullptr;
    void*   cuda_yuv_ptr = nullptr;
    uchar3* cuda_rgb_ptr = nullptr;

    cudaHostAlloc(&host_yuv_ptr, 2 * numPixel, cudaHostAllocPortable);
    cudaHostAlloc(&host_rgb_ptr, 3 * numPixel, cudaHostAllocPortable);
    cudaMalloc(&cuda_yuv_ptr, 2 * numPixel);
    cudaMalloc(&cuda_rgb_ptr, 3 * numPixel);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(host_yuv_ptr, bytestream.data(), bytestream.size(), cudaMemcpyHostToHost, stream);

    size_t sum = 0;
    Watch  watch;

    for (int i = 0; i < 1000; i++) {
        cudaMemcpyAsync(cuda_yuv_ptr, host_yuv_ptr, bytestream.size(), cudaMemcpyHostToDevice, stream);

        const uint8_t* d_yPlane  = reinterpret_cast<uint8_t*>(cuda_yuv_ptr);
        const uint8_t* d_uvPlane = reinterpret_cast<uint8_t*>(cuda_yuv_ptr) + width * height;

        cuda::image::yuv2rgb(d_yPlane, d_uvPlane, cuda_rgb_ptr, imgSize, false, stream);
        cudaMemcpyAsync(host_rgb_ptr, cuda_rgb_ptr, 3 * numPixel, cudaMemcpyDeviceToHost, stream);

        sum += reinterpret_cast<uchar3*>(host_rgb_ptr)[0].x; // Sum the blue channel of the first pixel
    }

    std::cout << "Total time for 1000 iterations: " << watch.elapsed() << " seconds\n";
    std::cout << "Sum of first pixel's blue channel over 1000 iterations: " << sum << "\n";

    cv::Mat cvimg(height, width, CV_8UC3, host_rgb_ptr);
    cv::cvtColor(cvimg, cvimg, cv::COLOR_RGB2BGR);
    cv::imshow("Cuda Image", cvimg);
    cv::waitKey(10000);

    return EXIT_SUCCESS;
}
