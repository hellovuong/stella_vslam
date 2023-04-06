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
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include "spdlog/spdlog.h"

namespace stella_vslam::data {
class keyframe;
class frame;
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
    virtual void add_keyframe(const std::shared_ptr<keyframe>& keyfrm) = 0;

    /**
     * Erase the keyframe from the database
     * @param keyfrm
     */
    virtual void erase_keyframe(const std::shared_ptr<keyframe>& keyfrm) = 0;

    /**
     * Clear the database
     */
    virtual void clear() = 0;

    /**
     * Compute representation BoW or global desc
     * @param keyframe
     */
    virtual void computeRepresentation(const std::shared_ptr<keyframe>& keyframe) = 0;

    /**
     * Compute representation BoW or global desc
     * @param frame
     */
    virtual void computeRepresentation(data::frame& frame, const cv::Mat& img) = 0;

    /**
     * Type of place_recognition_t
     */
    place_recognition_t database_type = place_recognition_type::BoW;

    static place_recognition_t load_vpr_type(const YAML::Node& node);

protected:
    //! mutex to access BoW database
    mutable std::mutex mtx_;
};

} // namespace stella_vslam::data

#endif // STELLA_VSLAM_DATA_BASE_DATABASE_H
