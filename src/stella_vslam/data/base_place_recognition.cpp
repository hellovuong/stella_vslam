//
// Created by vuong on 3/18/23.
//

#include "stella_vslam/data/keyframe.h"

#include "base_place_recognition.h"
#include "spdlog/spdlog.h"

namespace stella_vslam::data {
base_place_recognition::~base_place_recognition() {
    clear();
    spdlog::debug("DESTRUCT: data::base_database");
}

void base_place_recognition::clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    keyfrms_in_node_.clear();
    keyfrms.clear();
    spdlog::info("clear keyframe database");
}

void base_place_recognition::add_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    switch (database_type) {
        case place_recognition_type::BoW:
            // Append keyframe to the corresponding index in keyframes_in_node_ list
            for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
                keyfrms_in_node_[node_id_and_weight.first].push_back(keyfrm);
            }
            break;
        case place_recognition_type::HF_Net:
            keyfrms.insert(keyfrm);
            break;
    }
}

void base_place_recognition::erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) {
    std::lock_guard<std::mutex> lock(mtx_);
    switch (database_type) {
        case place_recognition_type::BoW:
            // Delete keyframe from the corresponding index in keyframes_in_node_ list
            for (const auto& node_id_and_weight : keyfrm->bow_vec_) {
                // first: node ID, second: weight
                if (!static_cast<bool>(keyfrms_in_node_.count(node_id_and_weight.first))) {
                    continue;
                }
                // Obtain keyframe which shares word
                auto& keyfrms_in_node = keyfrms_in_node_.at(node_id_and_weight.first);

                // std::list::erase only accepts iterator
                for (auto itr = keyfrms_in_node.begin(), lend = keyfrms_in_node.end(); itr != lend; itr++) {
                    if (keyfrm->id_ == (*itr)->id_) {
                        keyfrms_in_node.erase(itr);
                        break;
                    }
                }
            }
            break;
        case place_recognition_type::HF_Net:
            if (keyfrms.count(keyfrm)) {
                keyfrms.erase(keyfrm);
            }
            break;
    }
}
float base_place_recognition::compute_score(const cv::Mat& global_desc_1, const cv::Mat& global_desc_2) {
    assert(not global_desc_1.empty());
    assert(not global_desc_2.empty());

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> eigen_global_desc_1(global_desc_1.ptr<float>(), global_desc_1.rows, global_desc_1.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> eigen_global_desc_2(global_desc_2.ptr<float>(), global_desc_2.rows, global_desc_2.cols);

    return std::max(0.f, 1.f - (eigen_global_desc_1 - eigen_global_desc_2).norm());
}

} // namespace stella_vslam::data