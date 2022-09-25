#include "Module_lane_detect_impl.h"
#include "Module_lane_detect.h"

#include "debug.h"


CModule_lane_detect::~CModule_lane_detect()
{
    delete ANY_POINTER_CAST(impl_, CModule_lane_detect_impl);
#if defined(ALG_DEBUG) || defined(AI_ALG_DEBUG)
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_lane_detect::init(const LaneDetectConfig& config)
{
    ANY_POINTER_CAST(impl_, CModule_lane_detect_impl)->init(config);
}

void CModule_lane_detect::process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType)
{
    ANY_POINTER_CAST(impl_, CModule_lane_detect_impl)->process(src, src_height, src_width, inputDataType);
}

const std::vector<PointInt>& CModule_lane_detect::get_result()
{
    return ANY_POINTER_CAST(impl_, CModule_lane_detect_impl)->get_result();
}
