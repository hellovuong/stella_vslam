//
// Created by vuong on 3/18/23.
//

#include "stella_vslam/data/keyframe.h"

#include "hf_net_database.h"

namespace stella_vslam::data {
std::vector<std::shared_ptr<keyframe>> hf_net_database::acquire_keyframes(const cv::Mat& global_desc,
                                                                          float min_score,
                                                                          const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject) {
    float best_score;
    auto scores = compute_scores(global_desc, min_score, best_score);
    min_score = best_score * 0.8f;

    std::unordered_set<std::shared_ptr<keyframe>> final_candidates;
    for (const auto& keyfrm_score : scores) {
        if (keyfrm_score.second < min_score or static_cast<bool>(keyfrms_to_reject.count(keyfrm_score.first))) {
            continue;
        }
        const auto keyfrm = keyfrm_score.first;
        final_candidates.insert(keyfrm);
    }

    return {final_candidates.begin(), final_candidates.end()};
}
std::unordered_map<std::shared_ptr<keyframe>, float> hf_net_database::compute_scores(const cv::Mat& global_desc, float min_score, float& best_score) const {
    std::unordered_map<std::shared_ptr<keyframe>, float> scores;
    best_score = min_score;
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& keyframe : keyfrms) {
        // Compute the distance of global descriptors
        auto score = base_place_recognition::compute_score(global_desc, keyframe->frm_obs_.global_descriptors_);
        if (score <= min_score) {
            continue;
        }
        best_score = std::max(score, best_score);
        scores[keyframe] = score;
    }

    return scores;
}
} // namespace stella_vslam::data