//
// Created by vuong on 3/18/23.
//

#ifndef STELLA_VSLAM_DATA_HF_NET_DATABASE_H
#define STELLA_VSLAM_DATA_HF_NET_DATABASE_H

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>

#include "stella_vslam/data/base_place_recognition.h"

namespace stella_vslam::data {

class hf_net_database : public base_place_recognition {
public:
    /**
     * Constructor
     */
    explicit hf_net_database()
        : base_place_recognition(place_recognition_type::HF_Net){};

    /**
     * Destructor
     */
    ~hf_net_database();

    /**
     * Acquire keyframes over score
     * @param global_desc
     * @param min_score Compute by covis keyfrms
     * @param keyfrms_to_reject
     * @return
     */
    std::vector<std::shared_ptr<keyframe>> acquire_keyframes(const cv::Mat& global_desc,
                                                             float min_score,
                                                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {});

protected:
    /**
     * Compute scores (scores_) between the query and the each of keyframes contained in the database
     * @param global_desc [in]
     * @param min_score [in]
     * @param best_score [in,out]
     * @return similarity scores between the query and the each of keyframes contained in the database (key: keyframes that similar with query keyframe, value: score)
     */
    std::unordered_map<std::shared_ptr<keyframe>, float>
    compute_scores(const cv::Mat& global_desc, float min_score, float& best_score) const;
};

} // namespace stella_vslam::data

#endif // STELLA_VSLAM_DATA_HF_NET_DATABASE_H
