//
// Created by vuong on 3/2/23.
//

#include "relocalizer_hloc.h"

namespace stella_vslam::module {

relocalizer_hloc::relocalizer_hloc(const std::shared_ptr<optimize::pose_optimizer>& pose_optimizer,
                                   const std::shared_ptr<hloc::hloc_database>& hloc_database)
    : pose_optimizer_(pose_optimizer), hlocDatabase_(hloc_database) {

}
bool relocalizer_hloc::relocalize(const std::shared_ptr<hloc_database>& hloc_database, data::frame& curr_frm, const cv::Mat& rgb) {
    // compute global desc
    cv::Mat curr_frm_global_desc;
    NetVLAD::Extract(rgb, curr_frm_global_desc);
    auto candidate_keyfrms = hloc_database->query(curr_frm_global_desc);

    if (candidate_keyfrms.empty())
        return false;

    //! compute hloc data
    std::vector<cv::Point2f> curr_frm_keypoints;
    std::vector<float> curr_frm_keypoint_scores;
    cv::Mat curr_frm_local_descriptors;
    stella_vslam::SuperPoint::Extract(rgb, curr_frm_keypoints, curr_frm_keypoint_scores, curr_frm_local_descriptors);

    auto best_candidate = candidate_keyfrms.front().first;

    height = rgb.rows;
    width = rgb.cols;

    auto ok = reloc_by_candidate(curr_frm, curr_frm_keypoints, curr_frm_keypoint_scores, curr_frm_local_descriptors,
                                 best_candidate->keyfrm_, best_candidate->point_2d_uv_,
                                 best_candidate->scores_, best_candidate->local_descriptors_);

    return false;
}
bool relocalizer_hloc::reloc_by_candidate(stella_vslam::data::frame& curr_frm,
                                          std::vector<cv::Point2f>& curr_frm_keypoints,
                                          std::vector<float>& curr_frm_keypoint_scores,
                                          cv::Mat& curr_frm_local_descriptors,
                                          std::shared_ptr<stella_vslam::data::keyframe>& candidate_keyfrm,
                                          std::vector<cv::Point2f>& candidate_keyfrm_keypoints,
                                          std::vector<float>& candidate_keyfrm_keypoint_scores,
                                          cv::Mat& candidate_keyfrm_local_desc) {
    std::vector<unsigned int> inlier_indices;
    std::vector<std::shared_ptr<data::landmark>> matched_landmarks;
    bool ok = relocalize_by_pnp_solver(curr_frm, curr_frm_keypoints, curr_frm_keypoint_scores, curr_frm_local_descriptors,
                                       candidate_keyfrm, candidate_keyfrm_keypoints,candidate_keyfrm_keypoint_scores, candidate_keyfrm_local_desc,
                                       inlier_indices, matched_landmarks);
    if (!ok) {
        return false;
    }
    // Set 2D-3D matches for the pose optimization
    curr_frm.erase_landmarks();
    for (const auto idx : inlier_indices) {
        // Set only the valid 3D points to the current frame
        curr_frm.add_landmark(matched_landmarks.at(idx), idx);
    }

    std::vector<bool> outlier_flags;
    ok = optimize_pose(curr_frm, candidate_keyfrm, outlier_flags);
    if (!ok) {
        return false;
    }

    std::set<std::shared_ptr<data::landmark>> already_found_landmarks;
    for (const auto idx : inlier_indices) {
        if (outlier_flags.at(idx)) {
            continue;
        }
        // Record the 3D points already associated to the frame keypoints
        already_found_landmarks.insert(matched_landmarks.at(idx));
    }

    ok = refine_pose(curr_frm, candidate_keyfrm, already_found_landmarks);
    if (!ok) {
        return false;
    }

    ok = refine_pose_by_local_map(curr_frm, candidate_keyfrm);
    return ok;
}
bool relocalizer_hloc::relocalize_by_pnp_solver(data::frame& curr_frm,
                                                std::vector<cv::Point2f>& curr_frm_keypoints,
                                                std::vector<float>& curr_frm_keypoint_scores,
                                                cv::Mat& curr_frm_local_descriptors,
                                                std::shared_ptr<data::keyframe>& candidate_keyfrm,
                                                std::vector<cv::Point2f>& key_frm_keypoints,
                                                std::vector<float>& key_frm_keypoint_scores,
                                                cv::Mat& candidate_keyfrm_local_desc,
                                                std::vector<unsigned int>& inlier_indices,
                                                std::vector<std::shared_ptr<data::landmark>>& matched_landmarks) const {
    std::vector<int> match_index;
    std::vector<float> match_score;
    SuperGlue::Match(curr_frm_keypoints, curr_frm_keypoint_scores, curr_frm_local_descriptors, height, width,
                     key_frm_keypoints, key_frm_keypoint_scores, candidate_keyfrm_local_desc, height, width,
                     match_index, match_score
    );


    return false;
}
} // namespace stella_vslam::module