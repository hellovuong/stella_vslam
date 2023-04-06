//
// Created by vuong on 3/18/23.
//
#include "stella_vslam/data/frame.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/hloc/hf_net.h"
#include "hf_net_database.h"
#include "spdlog/spdlog.h"

namespace stella_vslam::data {
hf_net_database::~hf_net_database() {
    spdlog::debug("DESTRUCT: data::hf_net_database");
    clear();
    delete hf_net_;
}
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

float hf_net_database::compute_score(const cv::Mat& global_desc_1, const cv::Mat& global_desc_2) {
    assert(not global_desc_1.empty());
    assert(not global_desc_2.empty());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> eigen_global_desc_1(global_desc_1.ptr<float>(), global_desc_1.rows, global_desc_1.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> eigen_global_desc_2(global_desc_2.ptr<float>(), global_desc_2.rows, global_desc_2.cols);

    return std::max(0.f, 1.f - (eigen_global_desc_1 - eigen_global_desc_2).norm());
}

std::unordered_map<std::shared_ptr<keyframe>, float> hf_net_database::compute_scores(const cv::Mat& global_desc, float min_score, float& best_score) const {
    std::unordered_map<std::shared_ptr<keyframe>, float> scores;
    best_score = min_score;
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& keyframe : keyfrms) {
        // Compute the distance of global descriptors
        auto score = compute_score(global_desc, keyframe->frm_obs_.global_descriptors_);
        if (score <= min_score) {
            continue;
        }
        best_score = std::max(score, best_score);
        scores[keyframe] = score;
    }

    return scores;
}
void hf_net_database::add_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (not keyfrms.count(keyfrm)) {
        keyfrms.insert(keyfrm);
    }
}
void hf_net_database::erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (keyfrms.count(keyfrm)) {
        keyfrms.erase(keyfrm);
    }
}
void hf_net_database::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    keyfrms.clear();
    spdlog::info("clear vpr database");
}
hloc::hf_net* hf_net_database::getHfNet() const {
    return hf_net_;
}
void hf_net_database::computeRepresentation(const std::shared_ptr<keyframe>& keyframe) {
    hf_net_->compute_global_descriptors(keyframe->img.clone(), keyframe->frm_obs_.global_descriptors_);
}
void hf_net_database::computeRepresentation(frame& frame, const cv::Mat& img) {
    hf_net_->compute_global_descriptors(img.clone(), frame.frm_obs_.global_descriptors_);
}
} // namespace stella_vslam::data