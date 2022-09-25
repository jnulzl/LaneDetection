//
// Created by jnulzl on 2020/6/20.
//

#ifndef YOLOV5_POST_PROCESS_H
#define YOLOV5_POST_PROCESS_H

#include <cstdint>
#include "data_type.h"

void ultra_fast_lane_detect_post_process(float* data, int batch, int height, int width,
                                         int griding_num,
                                         const int* row_anchor,
                                         int cls_num_per_lane,
                                         int net_height, int net_width,
                                         int img_height, int img_width,
                                         std::vector<PointInt>& res);

    template <typename Dtype>
    void Permute(const Dtype* bottom_data, const std::vector<int>& bottom_data_shape, const std::vector<int>& permute_order,
                 const int num_axes, Dtype* top_data, std::vector<int>& top_data_shape);

    template <typename Dtype>
    void Sigmod(Dtype* bottom_data, const int bottom_data_num);
#endif //YOLOV5_POST_PROCESS_H
