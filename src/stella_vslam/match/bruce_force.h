//
// Created by vuong on 3/9/23.
//

#ifndef STELLA_VSLAM_BRUCE_FORCE_H
#define STELLA_VSLAM_BRUCE_FORCE_H

#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "stella_vslam/match/super_glue.h"

namespace stella_vslam {
namespace data {
class landmark;
class frame;
class keyframe;
}
namespace match {
class bruce_force {
public:
    static size_t match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                        std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm);
    static size_t match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                        std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                        std::vector<cv::DMatch>& best_matches);
    const float TH_HIGH = 0.75;
    constexpr static const float TH_LOW = 0.6;
};
class sg_matcher {
public:
    explicit sg_matcher(const YAML::Node& node);
    size_t match(const std::shared_ptr<data::keyframe>& keyfrm, data::frame& frm,
                        std::vector<std::shared_ptr<data::landmark>>& matched_lms_in_frm,
                        std::vector<cv::DMatch>& best_matches);

    std::shared_ptr<SuperGlue> superGlue;
};
} // namespace match
} // namespace stella_vslam

#endif // STELLA_VSLAM_BRUCE_FORCE_H
