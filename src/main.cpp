#include "corekit/utils/filemgr.hpp"
#include "yolo/core.hpp"

#include <csignal>
#include <cstdio>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

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
    uchar3* host_rgb_ptr = nullptr;
    uchar2* cuda_yuv_ptr = nullptr;
    uchar3* cuda_rgb_ptr = nullptr;

    cudaHostAlloc(&host_yuv_ptr, numPixel * sizeof(uchar2), cudaHostAllocPortable);
    cudaHostAlloc(&host_rgb_ptr, numPixel * sizeof(uchar3), cudaHostAllocPortable);
    cudaMalloc(&cuda_yuv_ptr, numPixel * sizeof(uchar2));
    cudaMalloc(&cuda_rgb_ptr, numPixel * sizeof(uchar3));

    JsonMap config = File::loadJson("/home/z0286456/Documents/cuda-visu/res/yolo.json");

    Path                     engine = "/home/z0286456/Documents/cuda-visu/res/models/tensorrt/desktop/yolo26n.engine";
    std::vector<std::string> classes;

    for (const auto& [key, name] : config["classes"].items()) {
        classes.push_back(name.get<std::string>());
    }

    yolo::Core::Settings settings = {
        .engine  = engine,
        .classes = classes,
    };

    yolo::Core model(settings);

    // cudaMemcpyAsync(host_yuv_ptr, bytestream.data(), bytestream.size(), cudaMemcpyHostToHost, model.stream);

    // cudaMemcpyAsync(cuda_yuv_ptr, host_yuv_ptr, bytestream.size(), cudaMemcpyHostToDevice, model.stream);
    // const uint8_t* d_yPlane  = reinterpret_cast<uint8_t*>(cuda_yuv_ptr);
    // const uint8_t* d_uvPlane = reinterpret_cast<uint8_t*>(cuda_yuv_ptr) + width * height;
    // cuda::image::yuv2rgb(d_yPlane, d_uvPlane, cuda_rgb_ptr, imgSize, false, model.stream);

    cv::VideoCapture cap("/dev/video0");

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat img;
    model.load();

    while (cap.isOpened() && cap.read(img)) {
        cv::resize(img, img, cv::Size(width, height));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        cudaMemcpyAsync(cuda_rgb_ptr, img.data, numPixel * sizeof(uchar3), cudaMemcpyHostToDevice, model.stream);
        model.process(cuda_rgb_ptr, cuda_rgb_ptr, imgSize, 0.25);
        cudaMemcpyAsync(host_rgb_ptr, cuda_rgb_ptr, numPixel * sizeof(uchar3), cudaMemcpyDeviceToHost, model.stream);
        cudaStreamSynchronize(model.stream);

        cv::Mat cvimg(height, width, CV_8UC3, host_rgb_ptr);
        cv::resize(cvimg, cvimg, cv::Size(1280, 720));
        cv::imshow("Cuda Image", cvimg);

        if (cv::waitKey(1) == 27) { // Exit on 'ESC' key
            break;
        }
    }

    model.unload();

    return EXIT_SUCCESS;
}
