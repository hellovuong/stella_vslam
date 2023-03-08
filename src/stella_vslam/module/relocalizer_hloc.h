//
// Created by vuong on 3/2/23.
//

#ifndef STELLA_VSLAM_RELOCALIZER_HLOC_H
#define STELLA_VSLAM_RELOCALIZER_HLOC_H

#include "stella_vslam/optimize/pose_optimizer.h"
#include "stella_vslam/hloc/hloc_database.h"
#include "stella_vslam/hloc/hloc.h"
#include "stella_vslam/data/frame.h"

namespace stella_vslam {

namespace data {
class frame;
} // namespace data

namespace module {
class relocalizer_hloc {
public:
    relocalizer_hloc(const std::shared_ptr<stella_vslam::optimize::pose_optimizer>& pose_optimizer,
                     const std::shared_ptr<stella_vslam::hloc::hloc_database>& hloc_database);

    //! Relocalize the specified frame
    bool relocalize(const std::shared_ptr<stella_vslam::hloc::hloc_database>& hloc_database, data::frame& curr_frm, const cv::Mat& rgb);

    bool reloc_by_candidate(data::frame& curr_frm,
                            std::vector<cv::Point2f>& curr_frm_keypoints,
                            std::vector<float>& curr_frm_keypoint_scores,
                            cv::Mat& curr_frm_local_descriptors,
                            std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                            std::vector<cv::Point2f>& key_frm_keypoints,
                            std::vector<float>& key_frm_keypoint_scores,
                            cv::Mat& candidate_keyfrm_local_desc);

    bool relocalize_by_pnp_solver(data::frame& curr_frm,
                                  std::vector<cv::Point2f>& curr_frm_keypoints,
                                  std::vector<float>& curr_frm_keypoint_scores,
                                  cv::Mat& curr_frm_local_descriptors,
                                  std::shared_ptr<data::keyframe>& candidate_keyfrm,
                                  std::vector<cv::Point2f>& key_frm_keypoints,
                                  std::vector<float>& key_frm_keypoint_scores,
                                  cv::Mat& candidate_keyfrm_local_desc,
                                  std::vector<unsigned int>& inlier_indices,
                                  std::vector<std::shared_ptr<data::landmark>>& matched_landmarks) const;
    bool optimize_pose(data::frame& curr_frm,
                       const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                       std::vector<bool>& outlier_flags) const;
    bool refine_pose(data::frame& curr_frm,
                     const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                     const std::set<std::shared_ptr<data::landmark>>& already_found_landmarks) const;
    bool refine_pose_by_local_map(data::frame& curr_frm,
                                  const std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm) const;

private:
    //! pose optimizer
    std::shared_ptr<optimize::pose_optimizer> pose_optimizer_ = nullptr;

    //! hloc database generate from hloc
    std::shared_ptr<hloc::hloc_database> hlocDatabase_ = nullptr;

    //! current frame image
    cv::Mat curr_img_;

    //! image size
    int height = {};
    int width = {};

    //! min matches threshold
    size_t min_num_bow_matches_ = 20;
};
} // namespace module
} // namespace stella_vslam

#endif // STELLA_VSLAM_RELOCALIZER_HLOC_H
