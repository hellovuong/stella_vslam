//
// Created by vuong on 2/28/23.
//

#include "hloc_database.h"

#include <utility>
namespace stella_vslam::hloc {
keyframe::keyframe(std::shared_ptr<data::keyframe> keyfrm)
    : keyfrm_(std::move(keyfrm)) {
    computeNew();
}
void keyframe::computeNew() {
    const cv::Mat img = keyfrm_->img.clone();
    // compute global descriptor
    cv::Mat global_descriptors;
    const auto tp_3 = std::chrono::steady_clock::now();
//    stella_vslam::NetVLAD::Extract(img, global_descriptors);
    const auto tp_4 = std::chrono::steady_clock::now();
    global_descriptors_ = global_descriptors.clone();

    const auto nv_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_4 - tp_3).count();
    spdlog::debug("NetVlad: {}", nv_time);
}
} // namespace stella_vslam::hloc