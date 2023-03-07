//
// Created by vuong on 2/28/23.
//

#include "hloc_database.h"

#include <utility>
stella_vslam::hloc::keyframe::keyframe(std::shared_ptr<data::keyframe> keyfrm)
    : keyfrm_(std::move(keyfrm)) {
    computeWindow();
    computeNew(point_2d_uv_, scores_);
}
void stella_vslam::hloc::keyframe::computeWindow() {
}
void stella_vslam::hloc::keyframe::computeNew(std::vector<cv::Point2f>& keypoints, std::vector<float>& scores) {
    // extract keypoint and compute local descriptor
    cv::Mat local_descriptors;
    const cv::Mat img = keyfrm_->img.clone();
    const auto tp_1 = std::chrono::steady_clock::now();
    //    stella_vslam::SuperPoint::Extract(img, keypoints, scores, local_descriptors);
    const auto tp_2 = std::chrono::steady_clock::now();
    local_descriptors_ = local_descriptors.clone();

    // compute global descriptor
    cv::Mat global_descriptors;
    const auto tp_3 = std::chrono::steady_clock::now();
    stella_vslam::NetVLAD::Extract(img, global_descriptors);
    const auto tp_4 = std::chrono::steady_clock::now();
    global_descriptors_ = global_descriptors.clone();

    const auto sp_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
    const auto nv_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_4 - tp_3).count();
    spdlog::debug("SP: {}", sp_time);
    spdlog::debug("NetVlad: {}", nv_time);
}
