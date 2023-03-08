//
// Created by vuong on 2/28/23.
//

#ifndef STELLA_VSLAM_HLOC_H
#define STELLA_VSLAM_HLOC_H

#include <torch/script.h>
#include <opencv2/core.hpp>

#include "spdlog/spdlog.h"

namespace stella_vslam {

#define SuperPointPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/SuperPoint_1024.pt"
#define NetVLADPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/NetVLAD.pt"
#define SuperGluePath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/NetVLAD.pt"
#define UltraPointPath "/home/vuong/Dev/prv/ws/openvins_ws/src/ov_hloc/support_files/weights/models/UltraPoint.pt"
class NetVLAD {
public:
    static NetVLAD& self();
    static void Extract(
        const cv::Mat& image,
        cv::Mat& desc);

private:
    torch::jit::script::Module model;
    NetVLAD();
    void IExtract(
        const cv::Mat& image,
        cv::Mat& desc);
};
} // namespace stella_vslam

#endif // STELLA_VSLAM_HLOC_H
