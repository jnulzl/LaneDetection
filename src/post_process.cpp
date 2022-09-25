//
// Created by jnulzl on 2020/6/20.
//
#include <vector>
#include <algorithm>

#include "post_process.h"

void get_not_equal_zero_num_each_col(float* data, int height, int width, std::vector<int>& not_equal_zero_num_each_col)
{
    not_equal_zero_num_each_col.resize(width, 0);
    for (int col = 0; col < width; ++col)
    {
        for (int idx = 0; idx < height; ++idx)
        {
            if(std::fabs(data[idx * width + col]) > 1e-5)
            {
                ++not_equal_zero_num_each_col[col];
            }
        }
    }
}

void ultra_fast_lane_detect_post_process(float* data, int batch, int height, int width,
                                         int griding_num,
                                         const int* row_anchor,
                                         int cls_num_per_lane,
                                         int net_height, int net_width,
                                         int img_height, int img_width,
                                         std::vector<PointInt>& res)
{
    float linSpace = static_cast<float>((net_width - 1.0f) / (griding_num - 1.0f));
    std::vector<int> not_equal_zero_num_each_col;
    for (int idb = 0; idb < batch; ++idb)
    {
        float* data_ptr = data + idb * height * width;
        get_not_equal_zero_num_each_col(data_ptr, height, width, not_equal_zero_num_each_col);
        for (int idx = 0; idx < width; ++idx)
        {
            if(not_equal_zero_num_each_col[idx] > 2)
            {
                for (int idy = 0; idy < height; ++idy)
                {
                    float tmp = data_ptr[idy * width + idx];
                    if(tmp > 0)
                    {
                        int x = tmp * linSpace * img_width / net_width - 1;
                        int y = img_height * (1.0f * row_anchor[cls_num_per_lane - 1 - idy] / net_height) - 1;
                        res.emplace_back(PointInt{x, y});
                    }
                }
            }
        }
    }
}

template <typename Dtype>
void Permute(const Dtype* bottom_data, const std::vector<int>& bottom_data_shape, const std::vector<int>& permute_order,
                    const int num_axes, Dtype* top_data, std::vector<int>& top_data_shape) {
    std::vector<int> old_steps(num_axes, 1);
    top_data_shape.resize(num_axes);
    for (int i = 0; i < num_axes; ++i) {
        if (i == num_axes - 1) {
            old_steps[i] = 1;
        } else {
            old_steps[i] = 1;
            for (int idx = i+1; idx < num_axes; ++idx) {
                old_steps[i] *= bottom_data_shape[idx];
            }
        }
        top_data_shape[i] = bottom_data_shape[permute_order[i]];
    }

    std::vector<int> new_steps(num_axes,1);
    for (int i = 0; i < num_axes; ++i) {
        if (i == num_axes - 1) {
            new_steps[i] = 1;
        } else {
            new_steps[i] = 1;
            for (int idx = i+1; idx < num_axes; ++idx) {
                new_steps[i] *= top_data_shape[idx];
            }
        }
    }

    int count = 1;
    for (int idx = 0; idx < num_axes; ++idx) {
        count *= bottom_data_shape[idx];
    }
    for (int i = 0; i < count; ++i) {
        int old_idx = 0;
        int idx = i;
        for (int j = 0; j < num_axes; ++j) {
            int order = permute_order[j];
            old_idx += (idx / new_steps[j]) * old_steps[order];
            idx %= new_steps[j];
        }
//        top_data[i] = (Dtype)(1.0) / (1 + std::exp(-bottom_data[old_idx]));
        top_data[i] = bottom_data[old_idx];
    }
}

template void Permute<float>(const float* bottom_data, const std::vector<int>& bottom_data_shape, const std::vector<int>& permute_order,
                                const int num_axes, float* top_data, std::vector<int>& top_data_shape);


template <typename Dtype>
void Sigmod(Dtype* bottom_data, const int bottom_data_num) {
    for (int idx = 0; idx < bottom_data_num; ++idx) {
        bottom_data[idx] = (Dtype)(1.0) / (1 + std::exp(-bottom_data[idx])); //sigmod
    }
}

template void Sigmod<float>(float* bottom_data, const int bottom_data_num);