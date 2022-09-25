#ifndef MODULE_LANE_DETECT_IMPL_H
#define MODULE_LANE_DETECT_IMPL_H

#include <string>
#include <vector>

#include "data_type.h"

class CModule_lane_detect_impl
{
public:
	CModule_lane_detect_impl();
	virtual ~CModule_lane_detect_impl() ;

    void init(const LaneDetectConfig &config);

	void process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);

	const std::vector<PointInt>& get_result();

protected:
    virtual void pre_process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);
    virtual void post_process();

    virtual void engine_init() = 0;
    virtual void engine_run() = 0;

protected:
    LaneDetectConfig config_;

    std::vector<float> data_in_;
    NetFloatTensor net_input_float_tensor_;

    std::vector<float> data_out_;
    NetFloatTensor net_output_float_tensor_;

    std::vector<uint8_t> src_resize_;

	std::vector<PointInt> points_;
    int img_height_;
    int img_width_;
};

#endif // MODULE_LANE_DETECT_IMPL_H

