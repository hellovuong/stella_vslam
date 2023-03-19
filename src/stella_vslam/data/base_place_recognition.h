//
// Created by vuong on 3/18/23.
//

#ifndef STELLA_VSLAM_DATA_BASE_DATABASE_H
#define STELLA_VSLAM_DATA_BASE_DATABASE_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include "bow_vocabulary_fwd.h"

namespace stella_vslam::data {
class keyframe;

enum class place_recognition_type {
    BoW,
    HF_Net
};
typedef place_recognition_type place_recognition_t;

class base_place_recognition {
public:
    /**
     * Constructor
     * @param type place_recognition_type
     */
    explicit base_place_recognition(place_recognition_t vpc_type)
        : database_type(vpc_type) {}

    /**
     * Destructor
     */
    virtual ~base_place_recognition();

    /**
     * Add a keyframe to the database
     * @param keyfrm
     */
    void add_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Erase the keyframe from the database
     * @param keyfrm
     */
    void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm);

    /**
     * Clear the database
     */
    void clear();

    /**
     * Type of place_recognition_t
     */
    place_recognition_t database_type = place_recognition_type::BoW;

    static float compute_score(const cv::Mat& global_desc_1, const cv::Mat& global_desc_2);

protected:
    //! mutex to access BoW database
    mutable std::mutex mtx_;
    //! BoW database
    std::unordered_map<unsigned int, std::list<std::shared_ptr<keyframe>>> keyfrms_in_node_{};
    //! hf database
    std::unordered_set<std::shared_ptr<keyframe>> keyfrms{};
};

} // namespace stella_vslam::data

#endif // STELLA_VSLAM_DATA_BASE_DATABASE_H
