#include "yolo/base.hpp"
#include "yolo/bbox.hpp"
#include "yolo/pose.hpp"
#include "yolo/segm.hpp"

#include <csignal>
#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/core/hal/interface.h>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>

using namespace corekit::cuda;
using namespace corekit::types;
using namespace corekit::utils;

int main(int argc, char* argv[])
{
    const size_t width    = 1920;
    const size_t height   = 1080;
    const size_t numPixel = width * height;
    const uint2  imgSize  = make_uint2(width, height);

    cv::VideoCapture cap("/dev/video0");

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return EXIT_FAILURE;
    }

    yolo::Base::Settings settings;
    settings.config = "/home/orinagx/Documents/trinity-visu/res/yolo.json";

    settings.engine = "/home/orinagx/Documents/trinity-visu/res/models/tensorRT/jetson/yolo26n-bbox.engine";
    yolo::BBox bbox(settings);

    settings.engine = "/home/orinagx/Documents/trinity-visu/res/models/tensorRT/jetson/yolo26n-pose.engine";
    yolo::Pose pose(settings);

    settings.engine = "/home/orinagx/Documents/trinity-visu/res/models/tensorRT/jetson/yolo26n-segm.engine";
    yolo::Segm segm(settings);

    bbox.load();
    pose.load();
    segm.load();

    while (cap.isOpened()) {
        cv::Mat in_img;

        if (!cap.read(in_img)) {
            std::cerr << "Error: Could not read frame from camera" << std::endl;
            break;
        }

        Image3U d_img = Image3U::fromCvMat(in_img);

        d_img = d_img.resize(make_uint2(width, height));
        bbox.process(d_img, d_img, 0.5);
        // pose.process(d_img, d_img, 0.5);
        // segm.process(d_img, d_img, 0.5);
        d_img = d_img.resize(make_uint2(1280, 720));

        cv::Mat out_img = d_img.toCvMat();

        cv::imshow("Cuda Image", out_img);
        if (cv::waitKey(1) == 27) {
            std::cerr << "Exit on interrupt" << std::endl;
            break;
        }
    };

    bbox.unload();
    pose.unload();
    segm.unload();

    return EXIT_SUCCESS;
}
