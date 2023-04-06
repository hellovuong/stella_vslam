//
// Created by vuong on 3/18/23.
//

#ifndef STELLA_VSLAM_DATA_HF_NET_DATABASE_H
#define STELLA_VSLAM_DATA_HF_NET_DATABASE_H

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>

#include "stella_vslam/data/base_place_recognition.h"
#include "stella_vslam/hloc/hf_net.h"

namespace stella_vslam::data {

class hf_net_database : public base_place_recognition {
public:
    /**
     * Constructor
     */
    explicit hf_net_database(hloc::hf_net* hfNet)
        : base_place_recognition(place_recognition_type::HF_Net),
          hf_net_(hfNet){};

    /**
     * Destructor
     */
    ~hf_net_database() override;

    /**
     * Add a keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm) override;

    /**
     * Erase the keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) override;

    void computeRepresentation(const std::shared_ptr<keyframe>& keyframe) override;

    /**
     * Clear the database
     */
    void clear() override;

    /**
     * Acquire keyframes over score
     * @param global_desc
     * @param min_score Compute by covis keyfrms
     * @param keyfrms_to_reject
     * @return
     */
    std::vector<std::shared_ptr<keyframe>> acquire_keyframes(const cv::Mat& global_desc,
                                                             float min_score = 0.0f,
                                                             const std::set<std::shared_ptr<keyframe>>& keyfrms_to_reject = {});
    /**
     * Compute score between 2 desc(s)
     * @param global_desc_1
     * @param global_desc_2
     * @return
     */
    static float compute_score(const cv::Mat& global_desc_1, const cv::Mat& global_desc_2);
    hloc::hf_net* getHfNet() const;

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

private:
    //! hf_net
    hloc::hf_net* hf_net_ = nullptr;
    //! hf database
    std::unordered_set<std::shared_ptr<keyframe>> keyfrms{};
};

} // namespace stella_vslam::data

#endif // STELLA_VSLAM_DATA_HF_NET_DATABASE_H
