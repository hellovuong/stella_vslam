//
// Created by vuong on 3/7/23.
//

#ifndef STELLA_VSLAM_FEATURE_BASE_EXTRACTOR_H
#define STELLA_VSLAM_FEATURE_BASE_EXTRACTOR_H

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace stella_vslam::feature {
//! Feature type
enum class feature_types {
    ORB,
    SuperPoint
};
typedef stella_vslam::feature::feature_types feature_type_t;

class base_extractor {
public:
    base_extractor() = default;
    virtual ~base_extractor() = default;
    //! Extract keypoints and each descriptor of them
    /**
     *
     * @param in_image
     * @param in_image_mask
     * @param keypts
     * @param out_descriptors
     */
    virtual void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                         std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors)
        = 0;

    //! Get model type as string
    [[nodiscard]] std::string get_feature_type_string() const { return {}; }
    //! Load model type from YAML
    static feature_type_t load_feature_type(const YAML::Node& yaml_node);

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;
    feature_type_t feature_type_ = {};

protected:
    //! mask for extract
    cv::Mat rect_mask_;
};
} // namespace stella_vslam::feature
#endif // STELLA_VSLAM_FEATURE_BASE_EXTRACTOR_H
