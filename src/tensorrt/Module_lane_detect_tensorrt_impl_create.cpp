#include "tensorrt/Module_lane_detect_tensorrt_impl.h"
#include "Module_lane_detect.h"

CModule_lane_detect::CModule_lane_detect(const std::string& engine_name)
{
    impl_ = new ALG_ENGINE_IMPL(lane_detect, tensorrt);
}