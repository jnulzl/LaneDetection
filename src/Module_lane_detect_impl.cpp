#include <assert.h>
#include <string.h>

#include "Module_lane_detect_impl.h"
#include "pre_process.h"
#include "post_process.h"
#include "debug.h"


CModule_lane_detect_impl::CModule_lane_detect_impl()
{

}
CModule_lane_detect_impl::~CModule_lane_detect_impl() 
{
#ifdef AI_ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_lane_detect_impl::init(const LaneDetectConfig& config)
{
	config_ = config;
    engine_init();

    net_input_float_tensor_.batch = 1;
    net_input_float_tensor_.channels= config_.net_inp_channels;
    net_input_float_tensor_.height = config_.net_inp_height;
    net_input_float_tensor_.width = config_.net_inp_width;
    net_input_float_tensor_.format = NetFloatTensor::DimensionType::NCHW;
    data_in_.resize(config_.net_inp_channels * config_.net_inp_width * config_.net_inp_height + 64);
    src_resize_.resize(data_in_.size());
    net_input_float_tensor_.data = data_in_.data();

}

void CModule_lane_detect_impl::pre_process(const uint8_t *src, int src_height, int src_width, InputDataType inputDataType)
{
    size_t num_channels;
    switch (inputDataType)
    {
        case InputDataType::IMG_BGR:
        case InputDataType::IMG_RGB:
            num_channels = 3;
            break;
        case InputDataType::IMG_GRAY:
            num_channels = 1;
            break;
        default:
            num_channels = 4;
    }
    size_t src_stride = num_channels * src_width;
    size_t des_stride = num_channels * config_.net_inp_width;

    memset(src_resize_.data(), 0, src_resize_.size());

    ai_utils_resize_with_affine(src, src_height, src_width, src_stride,
                                src_resize_.data(), config_.net_inp_height, config_.net_inp_width,
                                des_stride,
                                num_channels, 0);

    color_normalize_scale_and_chw(src_resize_.data(), config_.net_inp_height, config_.net_inp_width,
                                  config_.net_inp_channels * config_.net_inp_width, config_.means, config_.scales,
                                  data_in_.data(), 1,
#if defined(USE_TFLITE) || defined(USE_TFLITEGPU)
            0);
#else
                                  1);
#endif
}

void CModule_lane_detect_impl::process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType)
{
    img_height_ = src_height;
    img_width_ = src_width;

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

    pre_process(src, src_height, src_width, inputDataType);

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("Preprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

    engine_run();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_run = std::chrono::system_clock::now();
    std::printf("Inference time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_run - end_time).count());
#endif

    post_process();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_post = std::chrono::system_clock::now();
    std::printf("Postprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time_post - end_time_run).count());
#endif
}

void CModule_lane_detect_impl::post_process()
{
    points_.clear();
    ultra_fast_lane_detect_post_process(net_output_float_tensor_.data,
                                        net_output_float_tensor_.batch,
                                        net_output_float_tensor_.height,
                                        net_output_float_tensor_.width,
                                        config_.griding_num,
                                        config_.row_anchor.data(),
                                        config_.cls_num_per_lane,
                                        config_.net_inp_height, config_.net_inp_width,
                                        img_height_, img_width_,
                                        points_
                                        );
}

const std::vector<PointInt>& CModule_lane_detect_impl::get_result()
{
    return points_;
}