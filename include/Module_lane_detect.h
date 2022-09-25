#ifndef MODULE_LANE_DETECT_H
#define MODULE_LANE_DETECT_H

#include <string>
#include <vector>
#include "data_type.h"
#include "alg_define.h"

class AIWORKS_PUBLIC CModule_lane_detect
{
public:
	CModule_lane_detect(const std::string& engine_name);
	~CModule_lane_detect();

	void init(const LaneDetectConfig& config);

    void process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);

    const std::vector<PointInt>& get_result();

private:
	AW_ANY_POINTER impl_;
};

#endif // MODULE_LANE_DETECT_H

