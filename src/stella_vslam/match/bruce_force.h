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

    size_t match_current_and_last_frames(data::frame& curr_frm, const data::frame& last_frm,
                                         std::vector<cv::DMatch>& best_matches) const;

    /**
     * For debug only
     * @param reloc_candidates
     */
    [[maybe_unused]] static void drawMatches(const cv::Mat& img1, const cv::Mat& img2,
                                             const std::vector<cv::KeyPoint>& undist_keypts_1, const std::vector<cv::KeyPoint>& undist_keypts_2,
                                             const std::vector<cv::DMatch>& matches_result);

    std::shared_ptr<SuperGlue> superGlue;

};
} // namespace match
} // namespace stella_vslam

#endif // STELLA_VSLAM_BRUCE_FORCE_H
