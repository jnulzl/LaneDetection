#include <iostream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Module_lane_detect.h"
#include "debug.h"

int main(int argc, char* argv[])
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::string weights_path;
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.push_back("input");
    output_names.push_back("output");

    std::string input_src;

    LaneDetectConfig config_tmp;
    // -----------------------------original arguments-----------------------------
//    float means_rgb[3] = {123.675f, 116.28f, 103.53f};
//    float scales_rgb[3] = {0.01712f, 0.01751f, 0.01743f}; // 1.0 / 255
    // -----------------------------Seeing is OK-----------------------------
    float means_rgb[3] = {0, 0, 0};
    float scales_rgb[3] = {0.0039215, 0.0039215, 0.0039215}; // 1.0 / 255

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;

    // -----------------------------culane-----------------------------
    config_tmp.griding_num = 200;
    config_tmp.row_anchor = {121, 131, 141, 150, 160, 170, 180, 189, 199,
                             209, 219, 228, 238, 248, 258, 267, 277, 287};
    // -----------------------------tusimple-----------------------------
//    config_tmp.griding_num = 100;
//    config_tmp.row_anchor = { 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
//                              116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
//                              168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
//                              220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
//                              272, 276, 280, 284};
    config_tmp.cls_num_per_lane = config_tmp.row_anchor.size();
    if(argc < 5)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path net_width net_height video_path"
                  << std::endl;
        return -1;
    }

    weights_path = std::string(argv[1]);
    deploy_path = std::string(argv[1]);
    config_tmp.net_inp_width = std::atoi(argv[2]);
    config_tmp.net_inp_height = std::atoi(argv[3]);
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.deploy_path = deploy_path;

    CModule_lane_detect lane_detect("tensorrt");
    lane_detect.init(config_tmp);

    input_src = argv[4];
    cv::VideoCapture cap(input_src);
    if (!cap.isOpened())
    {
        cap.open(0);

        if (!cap.isOpened())
        {
            std::cout << "Unable open video/camera " << input_src << std::endl;
            return -1;
        }
    }
    long frame_id = 0;
#ifdef __SHOW__
    std::string win_name = "Demo";
    cv::namedWindow(win_name, 0);
#endif
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (!frame.data)
        {
            break;
        }
        cv::Mat img_origin = frame.clone();
        cv::Mat img_show = frame.clone();
        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        lane_detect.process(frame.data, frame.rows, frame.cols);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const std::vector<PointInt>& res = lane_detect.get_result();
        std::cout << "frame_id:" << frame_id << " Detected obj num : " <<  res.size() << " TensorRT process each frame time = " << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP1 - startTP).count() << " ms" << std::endl;
        // show result
        for (size_t idx = 0; idx < res.size(); idx++)
        {
            int xmin    = res[idx].x;
            int ymin    = res[idx].y;
            cv::circle(img_show, cv::Point2i(xmin, ymin),2, cv::Scalar(255, 0, 0), 2);
        }
#ifdef __SHOW__
        cv::putText(img_show, "Frame : " + std::to_string(frame_id),
                    {20, 50},2, 1, {0, 255, 0}
                    );
        cv::imshow(win_name, img_show);
        if(27 == cv::waitKey(1))
            break;
#else
        cv::imwrite("res/" + std::to_string(frame_id) + ".jpg",img_show);
        if(frame_id > 100)
        {
            break;
        }
#endif
        frame_id++;
    }
    cap.release();
    return 0;
}
